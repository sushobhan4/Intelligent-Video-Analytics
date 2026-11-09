"""
================================================================================
Neural CCTV - Autonomous Pipeline Orchestrator
================================================================================
Coordinates:
  1. Ultra-fast hardware-accelerated motion event clipping (700-1000+ FPS)
  2. Thumbnail generation for instant UI display
  3. Sequential VLM analysis via Qwen3-VL-4B-Instruct (4-bit NF4 + SDPA)
  4. Real-time Google VideoPrism 768-d video-to-vector embedding + persistent ChromaDB indexing
================================================================================
"""

import os
import sys
import time
import math
import subprocess
import threading
import traceback
from datetime import datetime
from typing import Dict, List, Optional, Callable, Any

# Ensure UTF-8 output on Windows consoles
if sys.platform == "win32":
    try:
        sys.stdout.reconfigure(encoding="utf-8")
        sys.stderr.reconfigure(encoding="utf-8")
    except Exception:
        pass

import cv2
import re
import numpy as np
import torch

from clip_motion_events import (
    get_video_info,
    scan_motion_events_fast,
    merge_and_pad_intervals,
    extract_clip_ffmpeg,
    format_time,
    format_time_filename
)
from test_qwen3_pipeline import (
    load_qwen3_model,
    analyze_video_hierarchical,
    parse_resolution
)
from clip_vector_db import CCTVVectorStore

BASE_DIR = os.path.dirname(os.path.abspath(__file__))
CLIPS_DIR = os.path.join(BASE_DIR, "data", "clips")
os.makedirs(CLIPS_DIR, exist_ok=True)


def extract_thumbnail_ffmpeg(video_path: str, thumb_path: str, timestamp_sec: float = 0.5) -> bool:
    """Extracts a high quality single JPEG frame from a video clip for UI thumbnail preview."""
    cmd = [
        "ffmpeg", "-y",
        "-ss", f"{timestamp_sec:.2f}",
        "-i", video_path,
        "-vframes", "1",
        "-q:v", "2",
        thumb_path
    ]
    try:
        res = subprocess.run(cmd, stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL, timeout=10)
        return res.returncode == 0 and os.path.exists(thumb_path) and os.path.getsize(thumb_path) > 100
    except Exception:
        return False


class CCTVAutonomousPipeline:
    """Manages the full end-to-end background pipeline."""

    def __init__(self, model_path: str = r"models\Qwen3-VL-4B-Instruct"):
        self.model_path = os.path.abspath(model_path)
        self.vector_store = CCTVVectorStore()

        # Shared VLM model instances (lazy-loaded once on demand)
        self._vlm_model = None
        self._vlm_processor = None
        self._model_lock = threading.Lock()

        # Pipeline runtime state
        self._stop_requested = False
        self._is_running = False
        self._state_lock = threading.Lock()

        # Rolling chronological incident memory for entity re-identification & continuity
        self.timeline_memory: List[Dict[str, Any]] = []

        self.current_state: Dict[str, Any] = {
            "status": "idle",  # "idle", "clipping", "analyzing", "completed", "error", "stopped"
            "video_path": "",
            "video_name": "",
            "stage": "Idle",
            "progress_pct": 0,
            "scan_fps": 0.0,
            "total_clips": 0,
            "processed_clips": 0,
            "active_clip": None,
            "elapsed_seconds": 0.0,
            "eta_seconds": 0.0,
            "error_message": None,
            "events": []
        }

    def _ensure_vlm_loaded(self):
        """Loads Qwen3-VL into GPU/RAM once with 4-bit quantization and SDPA."""
        with self._model_lock:
            if self._vlm_model is None or self._vlm_processor is None:
                print(f"[*] Initializing Qwen3-VL-4B-Instruct from: {self.model_path}")
                self._vlm_model, self._vlm_processor = load_qwen3_model(
                    model_path=self.model_path,
                    gpu_memory="3200MiB",
                    cpu_memory="24GiB"
                )
                print("[+] Qwen3-VL Model loaded into memory successfully.")

    def get_prior_context_summary(self, max_events: int = 5) -> str:
        """Formats the most recent chronological events into a concise forensic timeline."""
        if not self.timeline_memory:
            return ""
        recent = self.timeline_memory[-max_events:]
        lines = []
        for ev in recent:
            num = ev.get("index", "?")
            time_str = ev.get("time_range") or ev.get("timestamp") or "Timeline event"
            title = ev.get("title", "Event")
            action = ev.get("action_summary", "").strip()
            if len(action) > 140:
                action = action[:137] + "..."
            lines.append(f"- Event #{num} [{time_str}]: {title}. Action: {action}")
        return "\n".join(lines)

    def get_state(self) -> Dict[str, Any]:
        with self._state_lock:
            return dict(self.current_state)

    def request_stop(self):
        """Signals the pipeline worker to stop gracefully."""
        self._stop_requested = True

    def run_pipeline(
        self,
        video_path: str,
        camera_id: str = "FRONT_PORCH_CAM_01",
        chunk_sec: float = 10.0,
        resolution: str = "640x360",
        pixel_threshold: int = 18,
        min_motion_percent: float = 0.5,
        status_callback: Optional[Callable[[Dict[str, Any]], None]] = None
    ):
        """
        Main execution worker (run in a background thread).
        Stage 1: Fast scan & lossless clipping
        Stage 2: Sequential VLM analysis + Nomic embedding & ChromaDB indexing
        """
        video_path = os.path.abspath(video_path)
        if not os.path.exists(video_path):
            raise FileNotFoundError(f"Input video not found: {video_path}")

        video_name = os.path.basename(video_path)
        video_base = os.path.splitext(video_name)[0]
        clip_out_dir = os.path.join(CLIPS_DIR, video_base)
        thumb_out_dir = os.path.join(clip_out_dir, "thumbnails")
        os.makedirs(clip_out_dir, exist_ok=True)
        os.makedirs(thumb_out_dir, exist_ok=True)

        self._stop_requested = False
        t_start = time.time()

        with self._state_lock:
            self._is_running = True
            self.current_state.update({
                "status": "clipping",
                "video_path": video_path,
                "video_name": video_name,
                "stage": "Stage 1: Scanning & Fast Lossless Clipping (CUVID/FFmpeg)",
                "progress_pct": 5,
                "scan_fps": 0.0,
                "total_clips": 0,
                "processed_clips": 0,
                "active_clip": None,
                "elapsed_seconds": 0.0,
                "eta_seconds": 0.0,
                "error_message": None,
                "events": []
            })

        if status_callback:
            status_callback(self.get_state())

        try:
            # ==================================================================
            # STAGE 1: Fast Motion Event Detection & Lossless Clipping
            # ==================================================================
            w, h, orig_fps, orig_frames, orig_duration = get_video_info(video_path)

            clip_cfg = {
                "input_video": video_path,
                "output_dir": clip_out_dir,
                "target_fps": 11.1,
                "pixel_threshold": pixel_threshold,
                "blur_kernel": 5,
                "min_motion_percent": min_motion_percent,
                "cooldown_sec": 1.5,
                "padding_before_sec": 3.0,
                "padding_after_sec": 3.0,
                "merge_gap_sec": 4.0,
                "downsample_size": [320, 180]
            }

            raw_events, scan_time, effective_fps = scan_motion_events_fast(
                video_path, clip_cfg, orig_duration, orig_fps, orig_frames
            )

            if self._stop_requested:
                self._handle_stopped(status_callback)
                return

            if not raw_events:
                with self._state_lock:
                    self.current_state.update({
                        "status": "completed",
                        "stage": "No motion events detected in video",
                        "progress_pct": 100,
                        "scan_fps": round(effective_fps, 1)
                    })
                if status_callback:
                    status_callback(self.get_state())
                return

            merged_intervals = merge_and_pad_intervals(raw_events, orig_duration, clip_cfg)
            total_clips = len(merged_intervals)

            with self._state_lock:
                self.current_state.update({
                    "scan_fps": round(effective_fps, 1),
                    "total_clips": total_clips,
                    "progress_pct": 20,
                    "stage": f"Extracted {total_clips} motion events. Preparing lossless clips..."
                })
            if status_callback:
                status_callback(self.get_state())

            extracted_clips_info = []
            for idx, item in enumerate(merged_intervals, 1):
                if self._stop_requested:
                    self._handle_stopped(status_callback)
                    return

                c_start = item["clip_start"]
                c_end = item["clip_end"]
                duration = c_end - c_start

                time_tag = f"{format_time_filename(c_start)}_to_{format_time_filename(c_end)}"
                clip_id = f"event_clip_{idx:02d}_{time_tag}"
                clip_filename = f"{clip_id}.mp4"
                thumb_filename = f"{clip_id}.jpg"

                clip_full_path = os.path.join(clip_out_dir, clip_filename)
                thumb_full_path = os.path.join(thumb_out_dir, thumb_filename)

                # Extract lossless MP4
                extract_clip_ffmpeg(video_path, c_start, c_end, clip_full_path)
                
                # Extract JPEG thumbnail from middle frame (where the motion subject is centered)
                mid_point = max(0.2, duration / 2.0)
                extract_thumbnail_ffmpeg(clip_full_path, thumb_full_path, timestamp_sec=mid_point)

                # URLs for frontend
                video_rel_url = f"/media/clips/{video_base}/{clip_filename}"
                thumb_rel_url = f"/media/clips/{video_base}/thumbnails/{thumb_filename}"

                clip_data = {
                    "clip_id": clip_id,
                    "index": idx,
                    "clip_path": clip_full_path,
                    "video_url": video_rel_url,
                    "thumbnail_url": thumb_rel_url,
                    "start_sec": round(c_start, 2),
                    "end_sec": round(c_end, 2),
                    "duration_sec": round(duration, 1),
                    "status": "pending",
                    "title": f"Motion Event #{idx:02d}",
                    "caption": f"Motion at {format_time(item['motion_a'])} -> {format_time(item['motion_b'])}",
                    "timestamp": "",
                    "description": "",
                    "action_summary": "",
                    "similarity": None
                }
                extracted_clips_info.append(clip_data)

            with self._state_lock:
                self.current_state.update({
                    "progress_pct": 30,
                    "events": extracted_clips_info
                })
            if status_callback:
                status_callback(self.get_state())

            # ==================================================================
            # STAGE 2: VLM Analysis & Simultaneous Vector Indexing
            # ==================================================================
            with self._state_lock:
                self.current_state.update({
                    "status": "analyzing",
                    "stage": "Stage 2: Qwen3-VL Forensic Analysis & Vector Indexing",
                    "progress_pct": 35
                })
            if status_callback:
                status_callback(self.get_state())

            # Ensure model is ready
            self._ensure_vlm_loaded()
            parsed_res = parse_resolution(resolution)
            self.timeline_memory = []

            for i, clip_info in enumerate(extracted_clips_info):
                if self._stop_requested:
                    self._handle_stopped(status_callback)
                    return

                clip_start_time = time.time()
                clip_info["status"] = "analyzing"

                with self._state_lock:
                    self.current_state.update({
                        "active_clip": clip_info["clip_id"],
                        "processed_clips": i,
                        "stage": f"Analyzing clip {i + 1} of {total_clips}: {clip_info['clip_id']}",
                        "progress_pct": int(35 + (i / total_clips) * 60)
                    })
                if status_callback:
                    status_callback(self.get_state())

                # Visual RAG Step 1: Pre-encode video into 768-d Google VideoPrism embedding
                clip_emb = self.vector_store.embedder.embed_video(clip_info["clip_path"])

                # Visual RAG Step 2: Query ChromaDB for prior events with high visual similarity
                rag_matches = self.vector_store.retrieve_prior_context_rag(
                    clip_embedding=clip_emb,
                    current_start_sec=clip_info["start_sec"],
                    camera_id=camera_id,
                    top_k=3,
                    min_cosine_threshold=0.88
                )
                prior_context_parts = []
                if rag_matches:
                    print(f"[*] [Visual RAG] Retrieved {len(rag_matches)} prior match(es) for {clip_info['clip_id']}:")
                    for rm in rag_matches:
                        print(f"    -> {rm['clip_id']} (similarity={rm['cosine_similarity']:.4f})")
                    prior_context_parts.append(self.vector_store.format_rag_context_for_prompt(rag_matches))
                else:
                    print(f"[*] [Visual RAG] No previous matches above threshold for {clip_info['clip_id']} (novel scene/entity).")

                recent_timeline = self.get_prior_context_summary(max_events=3)
                if recent_timeline:
                    prior_context_parts.append(f"RECENT CHRONOLOGICAL INCIDENTS (Immediate Prior Timeline):\n{recent_timeline}")

                prior_context = "\n\n".join(prior_context_parts)

                # Visual RAG Step 3: Run VLM Hierarchical Chunk Analysis + Master Synthesis with retrieved context
                forensic_report = analyze_video_hierarchical(
                    model=self._vlm_model,
                    processor=self._vlm_processor,
                    video_source=clip_info["clip_path"],
                    chunk_duration_sec=chunk_sec,
                    target_size=parsed_res,
                    max_new_tokens=4096,
                    prior_context=prior_context
                )

                # Parse structured fields
                parsed = self.vector_store.parse_forensic_report(forensic_report)
                clip_info["title"] = parsed["title"] or f"Event #{clip_info['index']:02d}"
                clip_info["caption"] = parsed["direction_of_motion"] or clip_info["caption"]
                clip_info["timestamp"] = parsed["timestamp"] or datetime.now().strftime("%Y-%m-%d %H:%M:%S")
                clip_info["action_summary"] = parsed["action_summary"] or ""
                clip_info["description"] = forensic_report
                clip_info["status"] = "indexed"

                # Update chronological timeline memory
                self.timeline_memory.append({
                    "clip_id": clip_info["clip_id"],
                    "index": clip_info.get("index", i + 1),
                    "time_range": f"{format_time(clip_info['start_sec'])} - {format_time(clip_info['end_sec'])}",
                    "timestamp": clip_info["timestamp"],
                    "title": clip_info["title"],
                    "action_summary": clip_info["action_summary"]
                })

                # Visual RAG Step 4: Index into ChromaDB reusing precomputed clip_emb (0 duplicate compute)
                self.vector_store.add_clip(
                    clip_id=clip_info["clip_id"],
                    forensic_report=forensic_report,
                    video_path=clip_info["clip_path"],
                    embedding=clip_emb,
                    metadata={
                        "camera_id": camera_id,
                        "clip_path": clip_info["clip_path"],
                        "video_url": clip_info["video_url"],
                        "thumbnail_url": clip_info["thumbnail_url"],
                        "start_sec": clip_info["start_sec"],
                        "end_sec": clip_info["end_sec"],
                        "duration_sec": clip_info["duration_sec"],
                        "source_video": video_name
                    }
                )

                # Dynamic ETA calculation
                clip_elapsed = time.time() - clip_start_time
                remaining_clips = total_clips - (i + 1)
                est_remaining_time = remaining_clips * clip_elapsed

                with self._state_lock:
                    self.current_state.update({
                        "processed_clips": i + 1,
                        "elapsed_seconds": round(time.time() - t_start, 1),
                        "eta_seconds": round(est_remaining_time, 1),
                        "progress_pct": int(35 + ((i + 1) / total_clips) * 60),
                        "events": extracted_clips_info
                    })
                if status_callback:
                    status_callback(self.get_state())

            # ==================================================================
            # STAGE 3: Completed
            # ==================================================================
            with self._state_lock:
                self._is_running = False
                self.current_state.update({
                    "status": "completed",
                    "stage": f"Pipeline Complete: {total_clips} clips analyzed & indexed in Vector DB.",
                    "progress_pct": 100,
                    "active_clip": None,
                    "elapsed_seconds": round(time.time() - t_start, 1),
                    "eta_seconds": 0.0
                })
            if status_callback:
                status_callback(self.get_state())

        except Exception as e:
            traceback.print_exc()
            with self._state_lock:
                self._is_running = False
                self.current_state.update({
                    "status": "error",
                    "stage": "Pipeline Error",
                    "error_message": str(e)
                })
            if status_callback:
                status_callback(self.get_state())

    def process_unprocessed_clips(
        self,
        camera_id: str = "FRONT_PORCH_CAM_01",
        chunk_sec: float = 10.0,
        resolution: str = "640x360",
        status_callback: Optional[Callable[[Dict[str, Any]], None]] = None
    ):
        """
        Finds all motion clips on disk in CLIPS_DIR that are not yet indexed in ChromaDB,
        and sequentially processes them with Qwen3-VL + Nomic Embed + ChromaDB.
        """
        raw_db = self.vector_store._collection.get(include=[])
        indexed_ids = set(raw_db["ids"]) if raw_db and raw_db.get("ids") else set()

        unprocessed_clips = []
        for root, dirs, files in os.walk(CLIPS_DIR):
            if os.path.basename(root) == "thumbnails":
                continue
            video_base = os.path.basename(root)
            for f in files:
                if not f.endswith(".mp4"):
                    continue
                clip_id = os.path.splitext(f)[0]
                if clip_id not in indexed_ids:
                    clip_path = os.path.join(root, f)
                    thumb_path = os.path.join(root, "thumbnails", f"{clip_id}.jpg")
                    m = re.match(r"event_clip_(\d+)_(\d+)m(\d+)s_to_(\d+)m(\d+)s", clip_id)
                    start_sec = 0.0
                    end_sec = 0.0
                    idx = 999
                    if m:
                        idx, m1, s1, m2, s2 = map(int, m.groups())
                        start_sec = float(m1 * 60 + s1)
                        end_sec = float(m2 * 60 + s2)

                    unprocessed_clips.append({
                        "clip_id": clip_id,
                        "idx": idx,
                        "clip_path": clip_path,
                        "thumb_path": thumb_path,
                        "video_base": video_base,
                        "start_sec": start_sec,
                        "end_sec": end_sec,
                        "duration_sec": max(0.0, end_sec - start_sec)
                    })

        unprocessed_clips.sort(key=lambda c: (c["start_sec"], c["idx"]))
        total_clips = len(unprocessed_clips)

        if total_clips == 0:
            with self._state_lock:
                self.current_state.update({
                    "status": "completed",
                    "stage": "All motion clips on disk are already processed and indexed in ChromaDB.",
                    "progress_pct": 100
                })
            if status_callback:
                status_callback(self.get_state())
            return

        self._stop_requested = False
        t_start = time.time()

        with self._state_lock:
            self._is_running = True
            self.current_state.update({
                "status": "analyzing",
                "video_path": "Batch Unprocessed",
                "video_name": f"{total_clips} Unprocessed Motion Events",
                "stage": f"Processing {total_clips} Unprocessed Clips with Qwen3-VL",
                "progress_pct": 5,
                "scan_fps": 0.0,
                "total_clips": total_clips,
                "processed_clips": 0,
                "active_clip": None,
                "elapsed_seconds": 0.0,
                "eta_seconds": 0.0,
                "error_message": None
            })

        if status_callback:
            status_callback(self.get_state())

        try:
            self._ensure_vlm_loaded()
            parsed_res = parse_resolution(resolution)

            for i, clip_info in enumerate(unprocessed_clips):
                if self._stop_requested:
                    self._handle_stopped(status_callback)
                    return

                clip_start_time = time.time()
                clip_id = clip_info["clip_id"]
                clip_path = clip_info["clip_path"]
                thumb_path = clip_info["thumb_path"]
                video_base = clip_info["video_base"]

                if not os.path.exists(thumb_path):
                    os.makedirs(os.path.dirname(thumb_path), exist_ok=True)
                    extract_thumbnail_ffmpeg(clip_path, thumb_path, timestamp_sec=max(0.2, clip_info["duration_sec"] / 2.0))

                with self._state_lock:
                    self.current_state.update({
                        "active_clip": clip_id,
                        "processed_clips": i,
                        "stage": f"Analyzing clip {i + 1} of {total_clips}: {clip_id}",
                        "progress_pct": int(5 + (i / total_clips) * 90)
                    })
                if status_callback:
                    status_callback(self.get_state())

                # Visual RAG Step 1: Pre-encode video into 768-d Google VideoPrism embedding
                clip_emb = self.vector_store.embedder.embed_video(clip_path)

                # Visual RAG Step 2: Query ChromaDB for prior events with high visual similarity
                rag_matches = self.vector_store.retrieve_prior_context_rag(
                    clip_embedding=clip_emb,
                    current_start_sec=clip_info["start_sec"],
                    camera_id=camera_id,
                    top_k=3,
                    min_cosine_threshold=0.88
                )
                prior_context_parts = []
                if rag_matches:
                    print(f"[*] [Visual RAG] Retrieved {len(rag_matches)} prior match(es) for {clip_id}:")
                    for rm in rag_matches:
                        print(f"    -> {rm['clip_id']} (similarity={rm['cosine_similarity']:.4f})")
                    prior_context_parts.append(self.vector_store.format_rag_context_for_prompt(rag_matches))
                else:
                    print(f"[*] [Visual RAG] No previous matches above threshold for {clip_id} (novel scene/entity).")

                recent_timeline = self.get_prior_context_summary(max_events=3)
                if recent_timeline:
                    prior_context_parts.append(f"RECENT CHRONOLOGICAL INCIDENTS (Immediate Prior Timeline):\n{recent_timeline}")

                prior_context = "\n\n".join(prior_context_parts)

                # Visual RAG Step 3: Run VLM with retrieved context
                forensic_report = analyze_video_hierarchical(
                    model=self._vlm_model,
                    processor=self._vlm_processor,
                    video_source=clip_path,
                    chunk_duration_sec=chunk_sec,
                    target_size=parsed_res,
                    max_new_tokens=4096,
                    prior_context=prior_context
                )

                parsed = self.vector_store.parse_forensic_report(forensic_report)
                thumb_url = f"/media/clips/{video_base}/thumbnails/{clip_id}.jpg"
                video_url = f"/media/clips/{video_base}/{clip_id}.mp4"

                self.timeline_memory.append({
                    "clip_id": clip_id,
                    "index": clip_info.get("idx", i + 1),
                    "time_range": f"{format_time(clip_info['start_sec'])} - {format_time(clip_info['end_sec'])}",
                    "timestamp": parsed.get("timestamp", ""),
                    "title": parsed.get("title", f"Event #{clip_info.get('idx', i + 1)}"),
                    "action_summary": parsed.get("action_summary", "")
                })

                # Visual RAG Step 4: Index into ChromaDB reusing precomputed clip_emb (0 duplicate compute)
                self.vector_store.add_clip(
                    clip_id=clip_id,
                    forensic_report=forensic_report,
                    video_path=clip_path,
                    embedding=clip_emb,
                    metadata={
                        "camera_id": camera_id,
                        "clip_path": clip_path,
                        "video_url": video_url,
                        "thumbnail_url": thumb_url,
                        "start_sec": clip_info["start_sec"],
                        "end_sec": clip_info["end_sec"],
                        "duration_sec": clip_info["duration_sec"],
                        "source_video": video_base
                    }
                )

                clip_elapsed = time.time() - clip_start_time
                remaining_clips = total_clips - (i + 1)
                est_remaining_time = remaining_clips * clip_elapsed

                with self._state_lock:
                    self.current_state.update({
                        "processed_clips": i + 1,
                        "elapsed_seconds": round(time.time() - t_start, 1),
                        "eta_seconds": round(est_remaining_time, 1),
                        "progress_pct": int(5 + ((i + 1) / total_clips) * 95)
                    })
                if status_callback:
                    status_callback(self.get_state())

            with self._state_lock:
                self._is_running = False
                self.current_state.update({
                    "status": "completed",
                    "stage": f"Completed batch analysis: {total_clips} clips analyzed and indexed in ChromaDB.",
                    "progress_pct": 100,
                    "active_clip": None,
                    "elapsed_seconds": round(time.time() - t_start, 1),
                    "eta_seconds": 0.0
                })
            if status_callback:
                status_callback(self.get_state())

        except Exception as e:
            traceback.print_exc()
            with self._state_lock:
                self._is_running = False
                self.current_state.update({
                    "status": "error",
                    "stage": "Batch Processing Error",
                    "error_message": str(e)
                })
            if status_callback:
                status_callback(self.get_state())

    def _handle_stopped(self, status_callback):
        with self._state_lock:
            self._is_running = False
            self.current_state.update({
                "status": "stopped",
                "stage": "Pipeline stopped by user."
            })
        if status_callback:
            status_callback(self.get_state())


# Singleton instance
pipeline_instance = CCTVAutonomousPipeline()
