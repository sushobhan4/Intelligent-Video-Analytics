import os
import sys
import time
import tempfile
import urllib.parse
import argparse
import requests
import cv2
import torch
import gc
from PIL import Image
from transformers import BitsAndBytesConfig, AutoModelForImageTextToText, AutoProcessor
from qwen_vl_utils import process_vision_info

# Ensure UTF-8 output on Windows
if sys.platform == "win32":
    try:
        sys.stdout.reconfigure(encoding="utf-8")
        sys.stderr.reconfigure(encoding="utf-8")
    except Exception:
        pass

# ==============================================================================
# Performance & CUDA Optimization Flags
# ==============================================================================
# Enable expandable segments to eliminate CUDA memory fragmentation
os.environ["PYTORCH_CUDA_ALLOC_CONF"] = "expandable_segments:True"

# Enable PyTorch C++ CUDA SDPA / FlashAttention & Mem-Efficient backends
if torch.cuda.is_available():
    torch.backends.cudnn.benchmark = True
    try:
        torch.backends.cuda.enable_flash_sdp(True)
        torch.backends.cuda.enable_mem_efficient_sdp(True)
        torch.backends.cuda.enable_math_sdp(True)
    except Exception:
        pass

# ==============================================================================
# Prompts: Structured Forensic Output (Title, Direction, Entities, Action)
# ==============================================================================

# Context injection block for chronological incident continuity & re-identification
CONTEXT_SECTION_TEMPLATE = """
VISUAL RAG PRIOR INCIDENT CONTEXT (High Visual Similarity Matches from Same Camera Timeline):
{prior_context}

TEMPORAL CONTINUITY, RE-IDENTIFICATION & STATE/PAYLOAD CHANGE DIRECTIVES:
- CAREFULLY CROSS-REFERENCE: Compare the moving subjects, vehicles, persons, or cargo in this clip with the prior incidents logged above.
- RE-IDENTIFICATION & STATE/CARGO CHANGE:
  * If a vehicle, person, or animal matches an entity from prior events (e.g. the blue auto-rickshaw, the cyclist with cart, a specific person), EXPLICITLY state this re-identification!
  * HIGHLIGHT ANY STATUS, CARGO OR TRAJECTORY CHANGE: If the entity was previously seen empty and is now loaded (e.g. loaded with cardboard boxes, goods, or passengers), or was previously arriving/entering and is now departing/leaving, explicitly state this change in TITLE, ENTITY BREAKDOWN, and FINAL ACTION SUMMARY!
    Example: "The blue three-wheeled auto-rickshaw (previously observed entering empty in Event #06) is now leaving the scene loaded with cardboard boxes in the rear compartment."
- NOVEL ENTITIES: If an entity has not appeared in prior events, introduce and describe them as a new subject.
"""

# Chunk-level prompt: captures present entities & motion direction; strictly avoids negative spam
FORENSIC_CHUNK_PROMPT = """
You are an expert CCTV Forensic Video Analyst. Analyze this 30-FPS video chunk with maximum forensic precision.

CRITICAL INSTRUCTIONS:
1. Timestamp: Note on-screen camera timestamp if visible.
2. POSITIVE IDENTIFICATION ONLY: Describe ONLY moving subjects (people, vehicles, animals, objects) and dynamic actions that are ACTUALLY PRESENT in the video. NEVER list absent entities (do NOT say "no animals", "no weapons", "no pedestrians", "no visible items", etc.).
3. Ignore Background: Do NOT describe stationary background elements (walls, road, stationary gates, trees, pavement).
4. For every moving entity present, describe in deep detail:
   - Entity Category & Type: Person (gender, build, clothing colors/types, footwear, carried items), Vehicle (type, body color, markings, driver/rider), Animal (species, fur pattern, color).
   - Chunk Motion Direction: Exact direction and movement trajectory during this chunk (e.g. entering from right side, moving toward gate, turning left, heading away from camera).
""".strip()

# Master Synthesis Prompt: merges all chunk directions into one overall trajectory; outputs timestamp ONLY ONCE; stops only on EOS
SYNTHESIS_PROMPT_TEMPLATE = """
You are a Lead CCTV Forensic Security Analyst. Below is a chronological log of observations extracted across consecutive 10-second 30-FPS chunks of a single CCTV event:
{context_section}
{timeline}

TASK: Synthesize the above chunk observations into a single, clean, cohesive forensic report.

CRITICAL FORMATTING RULES:
Output MUST strictly follow this exact structure and field names:

TIMESTAMP: <On-screen camera timestamp if visible, stated EXACTLY ONCE>
TITLE: <A short, concise one-line summary of the entire clip, noting entity continuity & cargo/status change if applicable>
OVERALL DIRECTION OF MOTION: <Single, unified trajectory combining directional movements across all chunks from start to finish>
CONSOLIDATED FORENSIC ENTITY BREAKDOWN:
<For every moving entity observed (person, vehicle, animal): deep visual description (colors, build, clothing, vehicle finish/type, carried/loaded items, and explicit cross-reference to earlier appearances & payload changes)>
FINAL ACTION SUMMARY:
<Coherent narrative of the entire dynamic event from start to finish, explaining how this action relates to earlier events if the subject is recurring (e.g. previously seen empty, now leaving loaded)>

STRICT PROHIBITIONS:
- Do NOT output semantic search tags or keyword lists.
- Do NOT output chunk headers (no 'Window [0.0s - 10.0s]', etc.).
- Do NOT list absent entities (never say 'no animals', 'no pedestrians', etc.).
- Do NOT write bureaucratic report headers (no 'Lead Analyst:', 'Report ID:', etc.).
- Output ONLY the final structured text.
""".strip()

# Single-pass prompt template used when video is <= chunk_duration_sec
SINGLE_PASS_PROMPT_TEMPLATE = """
You are an expert CCTV Forensic Video Analyst. Analyze this 30-FPS video clip with maximum forensic precision.
{context_section}
CRITICAL FORMATTING RULES:
Output MUST strictly follow this exact structure and field names:

TIMESTAMP: <On-screen camera timestamp if visible, stated EXACTLY ONCE>
TITLE: <A short, concise one-line summary of the entire clip, noting entity continuity & cargo/status change if applicable>
OVERALL DIRECTION OF MOTION: <Single, unified trajectory detailing overall movement from point of origin to destination/exit>
CONSOLIDATED FORENSIC ENTITY BREAKDOWN:
<For every moving entity present (person, vehicle, animal): deep visual description including gender, build, clothing, vehicle finish/type, carried/loaded items, and explicit cross-reference to earlier events & payload changes>
FINAL ACTION SUMMARY:
<Coherent narrative of the entire dynamic event from start to finish, explaining how this action relates to or follows previous events if the subject is recurring (e.g. previously seen empty, now leaving loaded)>

STRICT PROHIBITIONS:
- POSITIVE IDENTIFICATION ONLY: Describe ONLY moving subjects and dynamic actions that are ACTUALLY PRESENT. Never describe absent entities (never say "no animals", "no weapons", etc.).
- Ignore Background: Do NOT describe stationary background elements (walls, road, stationary gates, trees, pavement).
- Do NOT output semantic search tags or keyword lists.
- Output ONLY the final structured text.
""".strip()

SINGLE_PASS_PROMPT = SINGLE_PASS_PROMPT_TEMPLATE.format(context_section="")


def parse_resolution(res_input):
    """Parses '640x360', '1280x720', '720p', or tuple into (width, height)."""
    if isinstance(res_input, (tuple, list)):
        return (int(res_input[0]), int(res_input[1]))
    res_str = str(res_input).lower().replace("p", "").strip()
    if "x" in res_str:
        w, h = res_str.split("x")
        return (int(w), int(h))
    elif "," in res_str:
        w, h = res_str.split(",")
        return (int(w), int(h))
    elif res_str == "720":
        return (1280, 720)
    elif res_str == "480":
        return (854, 480)
    elif res_str == "360":
        return (640, 360)
    elif res_str == "270":
        return (480, 270)
    return (640, 360)


def load_qwen3_model(
    model_path=r"models\Qwen3-VL-4B-Instruct",
    gpu_memory="3200MiB",
    cpu_memory="24GiB",
    offload_folder="offload"
):
    """
    Loads Qwen3-VL with compiled SDPA attention and hybrid VRAM/RAM overflow mapping.
    If attention or weights exceed GPU capacity, layers seamlessly offload to System RAM.
    """
    model_path = os.path.abspath(model_path)
    print(f"[*] Loading Qwen3-VL from: {model_path}")
    print(f"[*] Hybrid Memory Limits: GPU VRAM Cap={gpu_memory} | System RAM Pool={cpu_memory}")

    os.makedirs(offload_folder, exist_ok=True)

    bnb_config = BitsAndBytesConfig(
        load_in_4bit=True,
        bnb_4bit_quant_type="nf4",
        bnb_4bit_compute_dtype=torch.float16,
        bnb_4bit_use_double_quant=True
    )

    processor = AutoProcessor.from_pretrained(model_path)

    # Hybrid device mapping: Caps VRAM and spills into RAM
    max_memory = {
        0: gpu_memory,
        "cpu": cpu_memory
    }

    model = AutoModelForImageTextToText.from_pretrained(
        model_path,
        quantization_config=bnb_config,
        device_map="auto",
        max_memory=max_memory,
        offload_folder=offload_folder,
        attn_implementation="sdpa"  # Compiled PyTorch C++ CUDA FlashAttention / Mem-Efficient kernels
    )

    return model, processor


def download_if_url(video_source):
    """Downloads remote HTTP/HTTPS video to a temporary file if needed."""
    parsed = urllib.parse.urlparse(video_source)
    if parsed.scheme in ["http", "https"]:
        print(f"[*] Downloading video from: {video_source}...")
        resp = requests.get(video_source, stream=True, timeout=30)
        resp.raise_for_status()
        suffix = os.path.splitext(parsed.path)[1] or ".mp4"
        temp_fd, temp_file = tempfile.mkstemp(suffix=suffix)
        with os.fdopen(temp_fd, "wb") as f:
            for chunk in resp.iter_content(chunk_size=1024 * 1024):
                if chunk:
                    f.write(chunk)
        return temp_file, True
    return video_source, False


def read_all_video_frames(video_path, target_size=(640, 360)):
    """Reads all actual frames sequentially to avoid inaccurate MP4 metadata headers."""
    cap = cv2.VideoCapture(video_path)
    fps = cap.get(cv2.CAP_PROP_FPS)
    if not fps or fps <= 0 or fps > 120:
        fps = 30.0

    target_w, target_h = target_size
    frames = []
    while cap.isOpened():
        ret, frame = cap.read()
        if not ret:
            break
        resized = cv2.resize(frame, (target_w, target_h), interpolation=cv2.INTER_AREA)
        rgb_frame = cv2.cvtColor(resized, cv2.COLOR_BGR2RGB)
        frames.append(Image.fromarray(rgb_frame))
    cap.release()

    return frames, fps


def analyze_single_chunk(model, processor, frames, prompt_to_use, chunk_label="", max_new_tokens=4096):
    """Runs Qwen3-VL on a single batch of 30-fps frames with hybrid execution."""
    if not frames or len(frames) == 0:
        return ""

    prompt_text = f"[{chunk_label}]\n" + prompt_to_use if chunk_label else prompt_to_use
    messages = [
        {
            "role": "user",
            "content": [
                {"type": "video", "video": frames, "fps": 30.0},
                {"type": "text", "text": prompt_text}
            ]
        }
    ]

    text = processor.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)
    image_inputs, video_inputs = process_vision_info(messages)

    # Route inputs to model's primary device
    primary_device = model.device if hasattr(model, "device") else "cuda:0"
    inputs = processor(
        text=[text],
        images=image_inputs,
        videos=video_inputs,
        padding=True,
        return_tensors="pt"
    ).to(primary_device)

    # Generate until true end-of-text (<|im_end|>)
    with torch.no_grad():
        generated_ids = model.generate(
            **inputs, 
            max_new_tokens=max_new_tokens,
            eos_token_id=processor.tokenizer.eos_token_id
        )
        generated_ids_trimmed = [
            out_ids[len(in_ids):] for in_ids, out_ids in zip(inputs.input_ids, generated_ids)
        ]
        response = processor.batch_decode(
            generated_ids_trimmed, skip_special_tokens=True, clean_up_tokenization_spaces=False
        )

    del inputs, image_inputs, video_inputs, generated_ids
    gc.collect()
    if torch.cuda.is_available():
        torch.cuda.empty_cache()

    return response[0].strip()


def synthesize_chunks(model, processor, chunk_results, max_new_tokens=4096, prior_context=None):
    """Combines chunk observations into a single consolidated final output with chronological event continuity."""
    valid_results = [res for res in chunk_results if res['description'].strip()]
    if not valid_results:
        return "No moving subjects detected across any chunk."

    timeline_str = "\n\n".join([
        f"--- Event Window {res['window']} ---\n{res['description']}" 
        for res in valid_results
    ])

    if prior_context and str(prior_context).strip():
        context_block = CONTEXT_SECTION_TEMPLATE.format(prior_context=str(prior_context).strip())
    else:
        context_block = ""

    prompt = SYNTHESIS_PROMPT_TEMPLATE.format(timeline=timeline_str, context_section=context_block)

    messages = [
        {
            "role": "user",
            "content": [{"type": "text", "text": prompt}]
        }
    ]
    text = processor.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)
    primary_device = model.device if hasattr(model, "device") else "cuda:0"
    inputs = processor(text=[text], padding=True, return_tensors="pt").to(primary_device)

    with torch.no_grad():
        out = model.generate(
            **inputs, 
            max_new_tokens=max_new_tokens,
            eos_token_id=processor.tokenizer.eos_token_id
        )
        res = processor.batch_decode(out[:, inputs.input_ids.shape[1]:], skip_special_tokens=True)[0]

    del inputs, out
    gc.collect()
    if torch.cuda.is_available():
        torch.cuda.empty_cache()
    return res.strip()


def analyze_video_hierarchical(
    model, 
    processor, 
    video_source, 
    chunk_duration_sec=10.0, 
    target_size=(640, 360),
    max_new_tokens=4096,
    prior_context=None
):
    """
    Main Hierarchical 30-FPS Video Analyzer:
    - Reads frames sequentially without relying on corrupt MP4 headers.
    - Slices video into max chunk_duration_sec chunks at full 30 FPS.
    - Captures movement direction per chunk.
    - Synthesizes all chunks into an overall consolidated direction of motion.
    - Injects chronological prior incident memory for entity re-identification.
    - Uses compiled SDPA attention and hybrid RAM spillover.
    - Timestamp appears strictly ONCE.
    - Returns ONLY the clean final text.
    """
    local_path, is_temp = download_if_url(video_source)

    try:
        all_frames, fps = read_all_video_frames(local_path, target_size=target_size)
        total_frames = len(all_frames)
        if total_frames == 0:
            return "Error: Could not extract any valid video frames from video."

        duration_sec = total_frames / fps
        frames_per_chunk = max(int(fps * chunk_duration_sec), 1)

        # Slice into chunks of max chunk_duration_sec (e.g. 10s @ 30 FPS = 300 frames)
        chunks = [
            all_frames[i : i + frames_per_chunk] 
            for i in range(0, total_frames, frames_per_chunk)
        ]
        # Filter out empty or degenerate chunks
        chunks = [c for c in chunks if len(c) >= 2]

        # Single Chunk (video is <= chunk_duration_sec)
        if len(chunks) <= 1:
            if prior_context and str(prior_context).strip():
                context_block = CONTEXT_SECTION_TEMPLATE.format(prior_context=str(prior_context).strip())
            else:
                context_block = ""
            prompt_to_use = SINGLE_PASS_PROMPT_TEMPLATE.format(context_section=context_block)

            final_text = analyze_single_chunk(
                model=model, 
                processor=processor, 
                frames=all_frames, 
                prompt_to_use=prompt_to_use, 
                chunk_label=f"0.0s - {duration_sec:.1f}s",
                max_new_tokens=max_new_tokens
            )
            return final_text

        # Multi-Chunk Processing
        chunk_results = []
        for i, chunk_frames in enumerate(chunks):
            start_f = i * frames_per_chunk
            end_f = start_f + len(chunk_frames)
            window_label = f"{start_f / fps:.1f}s - {end_f / fps:.1f}s"

            desc = analyze_single_chunk(
                model=model, 
                processor=processor, 
                frames=chunk_frames, 
                prompt_to_use=FORENSIC_CHUNK_PROMPT, 
                chunk_label=window_label,
                max_new_tokens=max_new_tokens
            )

            if desc.strip():
                chunk_results.append({
                    "chunk_idx": i + 1,
                    "window": window_label,
                    "description": desc
                })

        # Synthesize all chunks into the final structured output
        final_output_text = synthesize_chunks(
            model=model, 
            processor=processor, 
            chunk_results=chunk_results,
            max_new_tokens=max_new_tokens,
            prior_context=prior_context
        )
        return final_output_text

    finally:
        if is_temp and os.path.exists(local_path):
            try:
                os.remove(local_path)
            except Exception:
                pass


def main():
    parser = argparse.ArgumentParser(description="Compiled Forensic CCTV Video Analyzer (Full 30 FPS + Hybrid Memory)")
    parser.add_argument("video_source", nargs="?", default=r"test\clips\event_clip_05_04m50s_to_04m59s.mp4",
                        help="Path to local video file or HTTP/HTTPS video URL")
    parser.add_argument("--chunk-sec", "--chunk-duration", dest="chunk_duration", type=float, default=10.0,
                        help="Maximum chunk duration in seconds at 30 FPS (default: 10.0)")
    parser.add_argument("--target-size", "--resolution", dest="target_size", default="640x360",
                        help="Target frame resolution e.g. '640x360', '854x480', '1280x720', or '480x270' (default: 640x360)")
    parser.add_argument("--gpu-memory", default="3200MiB",
                        help="Maximum GPU VRAM allocated before overflowing to System RAM (default: 3200MiB)")
    parser.add_argument("--cpu-memory", default="24GiB",
                        help="Maximum System RAM allocation for hybrid overflow (default: 24GiB)")
    parser.add_argument("--model-path", default=r"models\Qwen3-VL-4B-Instruct",
                        help=r"Path to local model weights (default: models\Qwen3-VL-4B-Instruct)")
    parser.add_argument("--max-new-tokens", type=int, default=4096,
                        help="Maximum new tokens generated per pass (default: 4096)")
    parser.add_argument("--no-index", action="store_true",
                        help="Skip indexing the final forensic report into ChromaDB vector store")
    args = parser.parse_args()

    parsed_res = parse_resolution(args.target_size)
    video_input = os.path.abspath(args.video_source) if not urllib.parse.urlparse(args.video_source).scheme else args.video_source

    print("=" * 70)
    print("🎥 FORENSIC CCTV VIDEO ANALYZER (COMPILED SDPA + HYBRID MEMORY)")
    print(f"• Video Source       : {video_input}")
    print(f"• Chunk Duration     : {args.chunk_duration}s @ 30 FPS")
    print(f"• Frame Resolution   : {parsed_res[0]}x{parsed_res[1]}")
    print(f"• GPU VRAM Limit     : {args.gpu_memory}")
    print(f"• System RAM Pool    : {args.cpu_memory}")
    print("=" * 70)

    model, processor = load_qwen3_model(
        model_path=args.model_path,
        gpu_memory=args.gpu_memory,
        cpu_memory=args.cpu_memory
    )

    t0 = time.time()
    final_result = analyze_video_hierarchical(
        model=model,
        processor=processor,
        video_source=video_input,
        chunk_duration_sec=args.chunk_duration,
        target_size=parsed_res,
        max_new_tokens=args.max_new_tokens
    )

    print("\n" + "=" * 70)
    print("📋 FINAL FORENSIC CCTV OUTPUT:")
    print("=" * 70)
    print(final_result)
    print("=" * 70)
    print(f"[*] Total analysis completed in {time.time() - t0:.1f}s")

    if not args.no_index and final_result:
        try:
            print("\n" + "=" * 70)
            print("💾 INDEXING FORENSIC CLIP INTO LOCAL CHROMADB VECTOR STORE")
            print("=" * 70)
            from clip_vector_db import CCTVVectorStore
            store = CCTVVectorStore()
            clip_name = os.path.splitext(os.path.basename(video_input))[0]
            store.add_clip(
                clip_id=clip_name,
                forensic_report=final_result,
                metadata={
                    "clip_path": video_input,
                    "analyzed_at": time.strftime("%Y-%m-%d %H:%M:%S")
                }
            )
            print(f"[+] Clip '{clip_name}' successfully indexed with Nomic embeddings in ChromaDB.")
            print(f"[+] Total clips now in database: {store.count()}")
            print("=" * 70)
        except Exception as e:
            print(f"[!] Warning: Vector database indexing failed: {e}")


if __name__ == "__main__":
    main()
