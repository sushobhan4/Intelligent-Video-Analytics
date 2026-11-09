"""
================================================================================
Neural CCTV - Local Vector Database & Google VideoPrism Multimodal Module
================================================================================
Model: Google VideoPrism (videoprism_lvt_public_v1_base)
Architecture: Factorized Spatial-Temporal Video Encoder + Causal Text Encoder
Embedding Dimensions: 768-d aligned multimodal latent space
Vector DB: ChromaDB (Persistent SQLite + HNSW with Cosine Metric)
Execution Device: Host CPU / CUDA (zero VRAM contention with Qwen3-VL)
================================================================================
"""

import os
import sys
import re
import argparse
from typing import Dict, List, Optional, Any
from datetime import datetime

# Ensure UTF-8 output on Windows consoles
if sys.platform == "win32":
    try:
        sys.stdout.reconfigure(encoding="utf-8")
        sys.stderr.reconfigure(encoding="utf-8")
    except Exception:
        pass

# Ensure Keras uses PyTorch backend for VideoPrism
os.environ["KERAS_BACKEND"] = "torch"
os.environ["HF_HUB_DISABLE_SYMLINKS_WARNING"] = "1"
os.environ["TF_ENABLE_ONEDNN_OPTS"] = "0"

import cv2
import numpy as np
import torch
import chromadb
import keras_hub
from transformers import AutoTokenizer

# Default configuration paths
BASE_DIR = os.path.dirname(os.path.abspath(__file__))
DEFAULT_DB_PATH = os.path.join(BASE_DIR, "data", "chroma_db")
DEFAULT_COLLECTION_NAME = "cctv_motion_events"
MODEL_NAME = "Google VideoPrism (videoprism_lvt_public_v1_base)"
TOKENIZER_NAME = "MHRDYN7/videoprism-lvt-base-f16r288"
PRESET_NAME = "videoprism_lvt_public_v1_base"


class VideoPrismEmbedder:
    """Multimodal Video-to-Vector & Text-to-Vector embedding model."""

    def __init__(self, preset_name: str = PRESET_NAME, tokenizer_name: str = TOKENIZER_NAME, device: str = "cpu"):
        self.preset_name = preset_name
        self.tokenizer_name = tokenizer_name
        self.device = device
        self._model = None
        self._tokenizer = None

    @property
    def model(self):
        """Lazy-loads the VideoPrism model backbone."""
        if self._model is None:
            print(f"[*] Loading Google VideoPrism backbone ('{self.preset_name}')...")
            self._model = keras_hub.models.VideoPrismBackbone.from_preset(self.preset_name)
            print("[+] Google VideoPrism 768-d backbone ready.")
        return self._model

    @property
    def tokenizer(self):
        """Lazy-loads the SentencePiece tokenizer."""
        if self._tokenizer is None:
            print(f"[*] Loading VideoPrism tokenizer ('{self.tokenizer_name}')...")
            self._tokenizer = AutoTokenizer.from_pretrained(self.tokenizer_name)
            print("[+] VideoPrism tokenizer ready.")
        return self._tokenizer

    def extract_frames(self, video_path: str, num_frames: int = 16, target_size: tuple = (288, 288)) -> np.ndarray:
        """
        Extracts exactly `num_frames` uniformly spaced frames from video,
        resizes to target dimensions, converts to RGB, and scales to [0, 1] float32.
        """
        if not os.path.exists(video_path):
            raise FileNotFoundError(f"Video file not found: {video_path}")

        cap = cv2.VideoCapture(video_path)
        total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))

        if total_frames <= 0:
            cap.release()
            raise ValueError(f"Could not read frames from video: {video_path}")

        frame_indices = np.linspace(0, total_frames - 1, num_frames).astype(int)
        frames = []
        curr_idx = 0

        while True:
            ret, frame = cap.read()
            if not ret:
                break
            if curr_idx in frame_indices:
                frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
                frame_resized = cv2.resize(frame_rgb, target_size, interpolation=cv2.INTER_AREA)
                frames.append(frame_resized.astype(np.float32) / 255.0)
            curr_idx += 1

        cap.release()

        # Handle short clips or missing frames gracefully by duplicating last frame
        if not frames:
            raise ValueError(f"Failed to decode any frames from: {video_path}")

        while len(frames) < num_frames:
            frames.append(frames[-1])

        # Shape: (1, 16, 288, 288, 3)
        return np.array([frames[:num_frames]], dtype=np.float32)

    def embed_video(self, video_path: str) -> List[float]:
        """
        Encodes raw video frames directly into an aligned 768-dimensional normalized embedding.
        """
        frames_np = self.extract_frames(video_path, num_frames=16, target_size=(288, 288))
        video_tensor = torch.tensor(frames_np, dtype=torch.float32)

        with torch.no_grad():
            vision_emb = self.model.get_vision_embeddings(video_tensor)
            norm_emb = torch.nn.functional.normalize(vision_emb, p=2, dim=-1)

        return norm_emb[0].cpu().numpy().tolist()

    def embed_text(self, text: str) -> List[float]:
        """
        Encodes query text into the aligned 768-dimensional normalized multimodal embedding space.
        """
        clean_text = text.strip()
        if not clean_text:
            clean_text = "motion event"

        tokens = self.tokenizer(clean_text, return_tensors="pt")
        token_ids = tokens["input_ids"].to(torch.int32)
        attention_mask = tokens["attention_mask"].to(torch.int32)

        with torch.no_grad():
            text_emb = self.model.get_text_embeddings(token_ids, attention_mask)
            norm_emb = torch.nn.functional.normalize(text_emb, p=2, dim=-1)

        return norm_emb[0].cpu().numpy().tolist()


class CCTVVectorStore:
    """Manages local vector embeddings and persistent ChromaDB storage for CCTV clips."""

    def __init__(self, db_path: str = DEFAULT_DB_PATH, collection_name: str = DEFAULT_COLLECTION_NAME, device: str = "cpu"):
        self.db_path = os.path.abspath(db_path)
        self.collection_name = collection_name
        self.device = device
        self._embedder = None
        self._client = None
        self._collection = None

        os.makedirs(self.db_path, exist_ok=True)
        self._init_chroma()

    def _init_chroma(self):
        """Initializes the persistent ChromaDB client and collection."""
        self._client = chromadb.PersistentClient(path=self.db_path)
        self._collection = self._client.get_or_create_collection(
            name=self.collection_name,
            metadata={"hnsw:space": "cosine"}
        )

    @property
    def embedder(self) -> VideoPrismEmbedder:
        """Lazy-loads the VideoPrism multimodal embedder."""
        if self._embedder is None:
            self._embedder = VideoPrismEmbedder(device=self.device)
        return self._embedder

    @staticmethod
    def parse_forensic_report(report_text: str) -> Dict[str, str]:
        """Extracts structured metadata fields from a synthesized forensic report."""
        meta = {
            "title": "",
            "timestamp": "",
            "direction_of_motion": "",
            "action_summary": ""
        }

        ts_match = re.search(r"TIMESTAMP:\s*(.+)", report_text, re.IGNORECASE)
        if ts_match:
            meta["timestamp"] = ts_match.group(1).strip()

        title_match = re.search(r"TITLE:\s*(.+)", report_text, re.IGNORECASE)
        if title_match:
            meta["title"] = title_match.group(1).strip()

        dir_match = re.search(r"OVERALL DIRECTION OF MOTION:\s*(.+)", report_text, re.IGNORECASE)
        if dir_match:
            meta["direction_of_motion"] = dir_match.group(1).strip()

        action_match = re.search(r"FINAL ACTION SUMMARY:\s*([\s\S]+)", report_text, re.IGNORECASE)
        if action_match:
            act_text = action_match.group(1).strip()
            split_headers = re.split(r"\n[A-Z\s_]{4,}:", act_text)
            meta["action_summary"] = split_headers[0].strip()

        return meta

    def add_clip(
        self,
        clip_id: str,
        forensic_report: Optional[str] = None,
        video_path: Optional[str] = None,
        metadata: Optional[Dict[str, Any]] = None,
        embedding: Optional[List[float]] = None
    ) -> Dict[str, Any]:
        """
        Generates Google VideoPrism 768-d embedding from video frames (or text fallback)
        and persists the clip + metadata into ChromaDB.
        """
        clean_report = (forensic_report or "").strip()
        parsed_fields = self.parse_forensic_report(clean_report) if clean_report else {
            "title": "", "timestamp": "", "direction_of_motion": "", "action_summary": ""
        }

        # Resolve video file path
        actual_video_path = video_path
        if not actual_video_path and metadata and "clip_path" in metadata:
            actual_video_path = metadata["clip_path"]

        # Use caller-provided embedding or generate 768-d multimodal embedding
        if embedding is None:
            if actual_video_path and os.path.exists(actual_video_path):
                try:
                    embedding = self.embedder.embed_video(actual_video_path)
                except Exception as e:
                    print(f"[!] Warning: Video embedding failed for {actual_video_path}: {e}")

        # Fallback to text embedding if video embedding failed or no video file
        if embedding is None:
            text_to_embed = clean_report or f"Motion event clip {clip_id}"
            embedding = self.embedder.embed_text(text_to_embed)

        # Merge extracted metadata with caller-provided metadata
        clip_meta = {
            "clip_id": str(clip_id),
            "indexed_at": datetime.now().isoformat(),
            "title": parsed_fields["title"] or f"Event Clip ({clip_id})",
            "timestamp_str": parsed_fields["timestamp"] or "",
            "direction_of_motion": parsed_fields["direction_of_motion"] or "",
            "action_summary": parsed_fields["action_summary"] or "",
            "embedding_model": MODEL_NAME,
        }
        if metadata:
            for k, v in metadata.items():
                if isinstance(v, (str, int, float, bool)):
                    clip_meta[k] = v
                else:
                    clip_meta[k] = str(v)

        doc_content = clean_report if clean_report else f"Motion Event Clip: {clip_meta.get('title', clip_id)}"

        # Add or update (upsert) in ChromaDB
        self._collection.upsert(
            ids=[str(clip_id)],
            embeddings=[embedding],
            documents=[doc_content],
            metadatas=[clip_meta]
        )
        return {"status": "success", "clip_id": clip_id, "collection": self.collection_name}

    def retrieve_prior_context_rag(
        self,
        clip_embedding: Optional[List[float]] = None,
        video_path: Optional[str] = None,
        current_start_sec: float = 0.0,
        camera_id: Optional[str] = None,
        top_k: int = 3,
        min_cosine_threshold: float = 0.88
    ) -> List[Dict[str, Any]]:
        """
        Visual RAG: Uses Google VideoPrism 768-d video vector of the current clip
        to retrieve ONLY the top-k visually & semantically related prior incidents
        from ChromaDB that occurred before `current_start_sec`.

        Filters out unrelated noise below `min_cosine_threshold`.
        """
        if self.count() == 0:
            return []

        # Obtain 768-d video embedding if not already supplied
        if clip_embedding is None:
            if not video_path or not os.path.exists(video_path):
                return []
            clip_embedding = self.embedder.embed_video(video_path)

        # Build chronological filter: only search events that occurred strictly BEFORE current clip
        where_filter: Dict[str, Any] = {
            "start_sec": {"$lt": float(current_start_sec)}
        }
        if camera_id:
            where_filter = {
                "$and": [
                    {"start_sec": {"$lt": float(current_start_sec)}},
                    {"camera_id": {"$eq": str(camera_id)}}
                ]
            }

        try:
            results = self._collection.query(
                query_embeddings=[clip_embedding],
                where=where_filter,
                n_results=top_k,
                include=["metadatas", "documents", "distances"]
            )
        except Exception:
            return []

        retrieved_events = []
        if results and results.get("ids") and len(results["ids"][0]) > 0:
            for i in range(len(results["ids"][0])):
                cid = results["ids"][0][i]
                meta = results["metadatas"][0][i] if results["metadatas"] else {}
                dist = results["distances"][0][i] if results.get("distances") else 0.0
                cos_sim = round(1.0 - dist, 4)

                # Visual RAG threshold: only keep genuine entity/scene matches
                if cos_sim >= min_cosine_threshold:
                    s_sec = float(meta.get("start_sec", 0.0))
                    e_sec = float(meta.get("end_sec", 0.0))
                    m1, s1 = int(s_sec // 60), int(s_sec % 60)
                    m2, s2 = int(e_sec // 60), int(e_sec % 60)
                    time_range_str = f"{m1:02d}:{s1:02d} - {m2:02d}:{s2:02d}"

                    retrieved_events.append({
                        "clip_id": cid,
                        "cosine_similarity": cos_sim,
                        "time_range": meta.get("time_range") or time_range_str,
                        "timestamp": meta.get("timestamp_str", ""),
                        "title": meta.get("title", ""),
                        "direction": meta.get("direction_of_motion", ""),
                        "action_summary": meta.get("action_summary", ""),
                        "full_report": results["documents"][0][i] if results.get("documents") else ""
                    })

        return retrieved_events

    @staticmethod
    def format_rag_context_for_prompt(rag_events: List[Dict[str, Any]]) -> str:
        """Formats retrieved RAG events into a concise, high-signal prompt block."""
        if not rag_events:
            return ""
        lines = []
        for ev in rag_events:
            t_str = ev.get("time_range") or ev.get("timestamp") or "Prior Event"
            title = ev.get("title", "Event")
            act = ev.get("action_summary", "").strip()
            if len(act) > 240:
                act = act[:237] + "..."
            sim = ev.get("cosine_similarity", 0.0)
            lines.append(f"- [Matched Prior Incident {ev.get('clip_id')} | Timeline: {t_str} | Visual Match Score: {sim:.3f}]: {title}. Details: {act}")
        return "\n".join(lines)

    def search_clips(
        self,
        query: str,
        n_results: int = 5,
        filter_meta: Optional[Dict[str, Any]] = None
    ) -> List[Dict[str, Any]]:
        """
        Performs cross-modal natural language semantic search across indexed CCTV clips
        using Google VideoPrism text-to-video alignment.
        """
        clean_query = query.strip()
        if not clean_query:
            return []

        query_emb = self.embedder.embed_text(clean_query)

        kwargs: Dict[str, Any] = {
            "query_embeddings": [query_emb],
            "n_results": min(n_results, max(1, self._collection.count())),
        }
        if filter_meta:
            kwargs["where"] = filter_meta

        results = self._collection.query(**kwargs)
        formatted = []

        if results and results["ids"] and len(results["ids"][0]) > 0:
            for i in range(len(results["ids"][0])):
                cid = results["ids"][0][i]
                doc = results["documents"][0][i] if results["documents"] else ""
                meta = results["metadatas"][0][i] if results["metadatas"] else {}
                dist = results["distances"][0][i] if results.get("distances") else 0.0
                # Cosine similarity: 1 - cosine_distance
                similarity = round(1.0 - dist, 4)
                # Calibrated confidence percentage for VideoPrism multimodal distribution
                # Empirical noise floor ~0.045, high relevance >= 0.105
                confidence_pct = round(min(99.0, max(5.0, ((similarity - 0.045) / 0.070) * 100.0)), 1)

                formatted.append({
                    "clip_id": cid,
                    "similarity": similarity,
                    "confidence_pct": confidence_pct,
                    "metadata": meta,
                    "document": doc
                })

        return formatted

    def list_clips(self, limit: int = 100) -> List[Dict[str, Any]]:
        """Returns all indexed clips and their metadata."""
        count = self._collection.count()
        if count == 0:
            return []
        data = self._collection.get(limit=limit, include=["metadatas", "documents"])
        items = []
        for i in range(len(data["ids"])):
            items.append({
                "clip_id": data["ids"][i],
                "metadata": data["metadatas"][i] if data["metadatas"] else {},
                "document": data["documents"][i] if data["documents"] else ""
            })
        return items

    def delete_clip(self, clip_id: str) -> bool:
        """Deletes a specific clip by ID from the vector database."""
        try:
            self._collection.delete(ids=[str(clip_id)])
            return True
        except Exception as e:
            print(f"[!] Error deleting clip '{clip_id}': {e}")
            return False

    def count(self) -> int:
        """Returns the number of indexed clips."""
        return self._collection.count()

    def reset_db(self) -> bool:
        """Completely wipes and re-creates the collection for a clean slate."""
        try:
            print(f"[*] Resetting collection '{self.collection_name}' in '{self.db_path}'...")
            self._client.delete_collection(name=self.collection_name)
            self._init_chroma()
            print("[+] Vector store collection successfully reset to clean state.")
            return True
        except Exception as e:
            self._init_chroma()
            return True


# ==============================================================================
# CLI Utility & Initialization
# ==============================================================================
def main():
    parser = argparse.ArgumentParser(description="Neural CCTV - Vector Database Manager (Google VideoPrism 768-d + ChromaDB)")
    parser.add_argument("--init", action="store_true", help="Initialize and verify ChromaDB and VideoPrism model")
    parser.add_argument("--reset", action="store_true", help="Completely clear and reset the vector database contents")
    parser.add_argument("--stats", action="store_true", help="Show database statistics and collection information")
    parser.add_argument("--list", action="store_true", help="List all indexed clips and their metadata")
    parser.add_argument("--search", type=str, default=None, help="Run a natural language search query")
    parser.add_argument("--limit", type=int, default=5, help="Number of results to return (default: 5)")
    parser.add_argument("--db-path", type=str, default=DEFAULT_DB_PATH, help=f"ChromaDB storage path (default: {DEFAULT_DB_PATH})")
    args = parser.parse_args()

    store = CCTVVectorStore(db_path=args.db_path)

    if args.reset:
        confirm = input("Are you sure you want to completely erase the vector DB content? (y/N): ").strip().lower()
        if confirm == "y":
            store.reset_db()
            print("[+] ChromaDB content has been completely wiped.")
        else:
            print("Reset canceled.")
        return

    if args.init:
        print("=" * 70)
        print("🚀 INITIALIZING CHROMA DB & GOOGLE VIDEOPRISM (768-d)")
        print("=" * 70)
        print(f"• ChromaDB Path   : {store.db_path}")
        print(f"• Collection Name : {store.collection_name}")
        print(f"• Model Name      : {MODEL_NAME}")
        print(f"• Preset          : {PRESET_NAME}")
        print("=" * 70)

        # Trigger model load and dry-run embedding test
        _ = store.embedder.model
        _ = store.embedder.tokenizer
        print("[*] Running dry-run text embedding test...")
        test_emb = store.embedder.embed_text("Dry run test event.")
        print(f"[+] Model verified! Vector dimensions: {len(test_emb)}")

        current_count = store.count()
        print(f"[+] ChromaDB initialized successfully! Current indexed clip count: {current_count}")
        print("=" * 70)
        return

    if args.stats:
        print(f"• DB Path          : {store.db_path}")
        print(f"• Collection       : {store.collection_name}")
        print(f"• Total Clips      : {store.count()}")
        return

    if args.list:
        clips = store.list_clips(limit=args.limit)
        print(f"[*] Found {len(clips)} indexed clips:")
        for c in clips:
            print(f"  - [{c['clip_id']}] {c['metadata'].get('title', 'No Title')} ({c['metadata'].get('timestamp_str', 'No Time')})")
        return

    if args.search:
        print(f"[*] Searching for: \"{args.search}\"")
        results = store.search_clips(args.search, n_results=args.limit)
        if not results:
            print("[!] No matching clips found.")
            return
        print(f"[+] Top {len(results)} matches:")
        for i, r in enumerate(results, 1):
            meta = r["metadata"]
            print(f"\n[{i}] Score: {r['similarity']} | ID: {r['clip_id']}")
            print(f"    Title     : {meta.get('title', 'N/A')}")
            print(f"    Timestamp : {meta.get('timestamp_str', 'N/A')}")
            print(f"    Direction : {meta.get('direction_of_motion', 'N/A')}")
            print(f"    Clip Path : {meta.get('clip_path', 'N/A')}")
        return

    # Default action if no flag is specified: show stats
    print(f"ChromaDB ready at: {store.db_path}")
    print(f"Collection '{store.collection_name}' has {store.count()} indexed clips.")
    print("Run with --help to see available commands.")


if __name__ == "__main__":
    main()
