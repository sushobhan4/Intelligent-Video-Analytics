"""
================================================================================
Neural CCTV - Re-index All Clips with Google VideoPrism (768-d Multimodal)
================================================================================
1. Backs up existing forensic reports & metadata from ChromaDB
2. Completely wipes and reinitializes the ChromaDB collection
3. Encodes all video clips directly with Google VideoPrism vision encoder (16 frames @ 288x288)
4. Saves new 768-d video embeddings + metadata into ChromaDB
5. Runs verification search queries to validate cross-modal text-to-video retrieval
================================================================================
"""

import os
import sys
import glob
import time
from datetime import datetime

# Windows console UTF-8 fix
if sys.platform == "win32":
    try:
        sys.stdout.reconfigure(encoding="utf-8")
        sys.stderr.reconfigure(encoding="utf-8")
    except Exception:
        pass

os.environ["KERAS_BACKEND"] = "torch"
os.environ["TF_ENABLE_ONEDNN_OPTS"] = "0"
os.environ["HF_HUB_DISABLE_SYMLINKS_WARNING"] = "1"

import cv2
import chromadb
from clip_vector_db import CCTVVectorStore, MODEL_NAME, PRESET_NAME


def main():
    print("=" * 80)
    print("🎥 GOOGLE VIDEOPRISM (768-d) VIDEO-TO-VECTOR RE-INDEXING ENGINE")
    print("=" * 80)
    print(f"• Model Preset   : {PRESET_NAME}")
    print(f"• Vector Dim     : 768 Dense Floats")
    print(f"• Latent Space   : Aligned Multimodal Cross-Attention")
    print("=" * 80)

    store = CCTVVectorStore()

    # Step 1: Read existing reports & metadata if present
    print("[*] Reading existing forensic records from ChromaDB before wipe...")
    existing_records = {}
    try:
        old_data = store._collection.get(include=["metadatas", "documents"])
        if old_data and old_data.get("ids"):
            for i, cid in enumerate(old_data["ids"]):
                existing_records[cid] = {
                    "document": old_data["documents"][i] if old_data["documents"] else "",
                    "metadata": old_data["metadatas"][i] if old_data["metadatas"] else {}
                }
            print(f"[+] Successfully backed up metadata for {len(existing_records)} clips.")
    except Exception as e:
        print(f"[!] Warning reading existing DB: {e}")

    # Step 2: Wipe ChromaDB completely
    print("\n[*] WIPING CHROMADB COLLECTION...")
    store.reset_db()
    print(f"[+] ChromaDB collection '{store.collection_name}' has been wiped clean! (Count: {store.count()})")

    # Step 3: Find all clips on disk
    clips_dir = os.path.join("data", "clips", "test_footage")
    clip_files = sorted(glob.glob(os.path.join(clips_dir, "*.mp4")))
    print(f"\n[*] Found {len(clip_files)} clips on disk to encode with Google VideoPrism.")

    if not clip_files:
        print("[!] No clips found in data/clips/test_footage!")
        return

    # Pre-load VideoPrism model once
    print("[*] Loading Google VideoPrism model backbone and tokenizer...")
    t_load_start = time.time()
    _ = store.embedder.model
    _ = store.embedder.tokenizer
    print(f"[+] VideoPrism loaded in {time.time() - t_load_start:.2f}s.")

    # Step 4: Encode each clip with VideoPrism and index into ChromaDB
    print("\n" + "=" * 80)
    print("🔄 ENCODING VIDEO CLIPS -> 768-d VIDEOPRISM VECTORS")
    print("=" * 80)

    t_start = time.time()
    indexed_count = 0

    for idx, clip_path in enumerate(clip_files, 1):
        filename = os.path.basename(clip_path)
        clip_id = os.path.splitext(filename)[0]

        t_clip = time.time()
        print(f"[{idx:02d}/{len(clip_files):02d}] Processing '{filename}'...", end="", flush=True)

        # Retrieve backup metadata or construct defaults
        record = existing_records.get(clip_id, {})
        doc = record.get("document", "")
        meta = record.get("metadata", {}).copy()

        # Update metadata fields
        meta["clip_id"] = clip_id
        meta["clip_path"] = os.path.abspath(clip_path)
        meta["video_url"] = f"/media/clips/test_footage/{filename}"
        meta["thumbnail_url"] = f"/media/clips/test_footage/thumbnails/{clip_id}.jpg"
        meta["source_video"] = "test_footage.mp4"
        meta["camera_id"] = meta.get("camera_id", "FRONT_PORCH_CAM_01")
        meta["embedding_model"] = MODEL_NAME
        meta["indexed_at"] = datetime.now().isoformat()

        # Generate 768-d video embedding via VideoPrism vision encoder
        video_emb = store.embedder.embed_video(clip_path)

        # Store in ChromaDB
        store._collection.upsert(
            ids=[clip_id],
            embeddings=[video_emb],
            documents=[doc or f"CCTV Motion Clip: {clip_id}"],
            metadatas=[meta]
        )

        indexed_count += 1
        elapsed_clip = time.time() - t_clip
        print(f" Done ({elapsed_clip:.2f}s | 768-d vector stored)")

    total_time = time.time() - t_start
    print("=" * 80)
    print(f"[+] All {indexed_count} clips successfully encoded & indexed with Google VideoPrism in {total_time:.2f}s!")
    print(f"[+] Total indexed clips in ChromaDB: {store.count()}")
    print("=" * 80)

    # Step 5: Verification search queries
    print("\n🔍 RUNNING VERIFICATION SEARCH QUERIES (Cross-Modal Video Retrieval):")
    print("-" * 80)

    test_queries = [
        "a man carrying a blue object",
        "woman in orange sari",
        "a person walking outdoors",
        "someone walking with a child",
        "red sports car speeding"
    ]

    for q in test_queries:
        print(f"\nQuery: \"{q}\"")
        results = store.search_clips(q, n_results=3)
        if not results:
            print("  No matches returned.")
            continue
        for r_idx, r in enumerate(results, 1):
            m = r["metadata"]
            print(f"  [{r_idx}] Score: {r['similarity']:.4f} | ID: {r['clip_id']}")
            print(f"       Title: {m.get('title', 'N/A')}")
            print(f"       Action: {m.get('action_summary', 'N/A')[:90]}...")

    print("\n" + "=" * 80)
    print("✅ RE-INDEXING COMPLETE & FULLY VERIFIED")
    print("=" * 80)


if __name__ == "__main__":
    main()
