"""
================================================================================
Neural CCTV - Forensic Intelligence & Semantic Search Server (FastAPI)
================================================================================
Endpoints:
  - Video upload & discovery (/api/videos, /api/upload)
  - Autonomous pipeline control (/api/pipeline/start, /api/pipeline/stop, /api/pipeline/status)
  - Real-time WebSocket event streaming (/ws/pipeline)
  - Semantic vector search & sorting (/api/clips, /api/clips/search)
  - System diagnostics & ChromaDB management (/api/system/status, /api/clips/reset)
================================================================================
"""

import os
import sys
import json
import time
import shutil
import asyncio
import threading
from typing import Dict, List, Optional, Any

# Ensure UTF-8 output on Windows consoles
if sys.platform == "win32":
    try:
        sys.stdout.reconfigure(encoding="utf-8")
        sys.stderr.reconfigure(encoding="utf-8")
    except Exception:
        pass

import torch
import psutil
import platform
import re
import statistics
from collections import Counter
from datetime import datetime
from fastapi import FastAPI, UploadFile, File, Form, Query, HTTPException, WebSocket, WebSocketDisconnect
from fastapi.responses import FileResponse, StreamingResponse
from fastapi.middleware.cors import CORSMiddleware
from fastapi.staticfiles import StaticFiles
from pydantic import BaseModel

from clip_vector_db import CCTVVectorStore, MODEL_NAME
from cctv_pipeline import pipeline_instance, CLIPS_DIR

BASE_DIR = os.path.dirname(os.path.abspath(__file__))
DATA_DIR = os.path.join(BASE_DIR, "data")
UPLOADS_DIR = os.path.join(DATA_DIR, "uploads")
TEST_DIR = os.path.join(BASE_DIR, "test")

for d in [DATA_DIR, UPLOADS_DIR, CLIPS_DIR, TEST_DIR]:
    os.makedirs(d, exist_ok=True)

app = FastAPI(title="Neural CCTV - Forensic & Semantic Video Search API")

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# ==============================================================================
# Dynamic Self-Healing Thumbnail Handlers
# ==============================================================================
@app.get("/media/clips/{video_base}/thumbnails/{thumb_filename}")
async def get_clip_thumbnail_dynamic(video_base: str, thumb_filename: str):
    """Serves thumbnail, or auto-generates middle frame on the fly if missing."""
    thumb_path = os.path.join(CLIPS_DIR, video_base, "thumbnails", thumb_filename)
    if not os.path.exists(thumb_path) or os.path.getsize(thumb_path) == 0:
        clip_name = os.path.splitext(thumb_filename)[0]
        mp4_path = os.path.join(CLIPS_DIR, video_base, f"{clip_name}.mp4")
        if os.path.exists(mp4_path):
            os.makedirs(os.path.dirname(thumb_path), exist_ok=True)
            cmd = ["ffmpeg", "-y", "-ss", "2.0", "-i", mp4_path, "-vframes", "1", "-q:v", "2", thumb_path]
            subprocess.run(cmd, stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)
    if os.path.exists(thumb_path):
        return FileResponse(thumb_path, media_type="image/jpeg")
    raise HTTPException(status_code=404, detail="Thumbnail not found")


@app.get("/media/clips/{video_base}/{thumb_filename}.jpg")
async def get_clip_root_thumbnail_fallback(video_base: str, thumb_filename: str):
    """Fallback if browser requests /media/clips/<video>/<clip>.jpg directly."""
    sub_path = os.path.join(CLIPS_DIR, video_base, "thumbnails", f"{thumb_filename}.jpg")
    if os.path.exists(sub_path):
        return FileResponse(sub_path, media_type="image/jpeg")
    root_path = os.path.join(CLIPS_DIR, video_base, f"{thumb_filename}.jpg")
    if os.path.exists(root_path):
        return FileResponse(root_path, media_type="image/jpeg")
    mp4_path = os.path.join(CLIPS_DIR, video_base, f"{thumb_filename}.mp4")
    if os.path.exists(mp4_path):
        os.makedirs(os.path.dirname(sub_path), exist_ok=True)
        cmd = ["ffmpeg", "-y", "-ss", "2.0", "-i", mp4_path, "-vframes", "1", "-q:v", "2", sub_path]
        subprocess.run(cmd, stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)
        if os.path.exists(sub_path):
            return FileResponse(sub_path, media_type="image/jpeg")
    raise HTTPException(status_code=404, detail="Thumbnail not found")


# Serve media directories
app.mount("/media/clips", StaticFiles(directory=CLIPS_DIR), name="clips")
app.mount("/media/uploads", StaticFiles(directory=UPLOADS_DIR), name="uploads")
app.mount("/media/test", StaticFiles(directory=TEST_DIR), name="test")

# Global Vector Store
vector_store = CCTVVectorStore()


# ==============================================================================
# Real-Time Telemetry & WebSocket / SSE Manager
# ==============================================================================
class ConnectionManager:
    def __init__(self):
        self.active_connections: List[WebSocket] = []
        self.sse_queues: List[asyncio.Queue] = []
        self._lock = asyncio.Lock()

    async def connect(self, websocket: WebSocket):
        await websocket.accept()
        async with self._lock:
            self.active_connections.append(websocket)

    async def disconnect(self, websocket: WebSocket):
        async with self._lock:
            if websocket in self.active_connections:
                self.active_connections.remove(websocket)

    async def connect_sse(self, queue: asyncio.Queue):
        async with self._lock:
            self.sse_queues.append(queue)

    async def disconnect_sse(self, queue: asyncio.Queue):
        async with self._lock:
            if queue in self.sse_queues:
                self.sse_queues.remove(queue)

    def has_listeners(self) -> bool:
        return len(self.active_connections) > 0 or len(self.sse_queues) > 0

    async def broadcast(self, message: dict):
        data_text = json.dumps(message)
        async with self._lock:
            for connection in list(self.active_connections):
                try:
                    await connection.send_text(data_text)
                except Exception:
                    if connection in self.active_connections:
                        self.active_connections.remove(connection)
            for q in list(self.sse_queues):
                try:
                    q.put_nowait(message)
                except Exception:
                    if q in self.sse_queues:
                        self.sse_queues.remove(q)

    def broadcast_threadsafe(self, message: dict):
        """Dispatches a broadcast onto the main asyncio loop from worker threads."""
        global main_loop
        if main_loop and main_loop.is_running():
            try:
                asyncio.run_coroutine_threadsafe(self.broadcast(message), main_loop)
            except Exception as e:
                print(f"[!] Warning broadcasting threadsafe: {e}")
        else:
            try:
                loop = asyncio.get_running_loop()
                loop.create_task(self.broadcast(message))
            except RuntimeError:
                pass


ws_manager = ConnectionManager()
main_loop: Optional[asyncio.AbstractEventLoop] = None


def pipeline_callback(state: Dict[str, Any]):
    """Thread-safe callback to broadcast pipeline state across WebSockets & SSE."""
    ws_manager.broadcast_threadsafe({"type": "pipeline_state", "data": state})
    if state.get("status") in ["clipping", "analyzing", "completed"] or state.get("processed_clips", 0) > 0:
        ws_manager.broadcast_threadsafe({"type": "clips_updated"})


# ==============================================================================
# System & Hardware Diagnostics (Apple System Information Style)
# ==============================================================================
@app.get("/api/system/status")
async def get_system_status():
    # GPU Diagnostics
    gpu_info = {
        "available": torch.cuda.is_available(),
        "name": "N/A",
        "vram_total_gb": 0.0,
        "vram_allocated_gb": 0.0,
        "vram_free_gb": 0.0,
        "vram_percent": 0.0,
        "device_capability": "N/A"
    }
    if torch.cuda.is_available():
        gpu_info["name"] = torch.cuda.get_device_name(0)
        total_mem = torch.cuda.get_device_properties(0).total_memory / (1024 ** 3)
        alloc_mem = torch.cuda.memory_allocated(0) / (1024 ** 3)
        free_mem = max(0.0, total_mem - alloc_mem)
        pct = round((alloc_mem / total_mem) * 100, 1) if total_mem > 0 else 0.0
        cap = torch.cuda.get_device_capability(0)
        gpu_info["vram_total_gb"] = round(total_mem, 2)
        gpu_info["vram_allocated_gb"] = round(alloc_mem, 2)
        gpu_info["vram_free_gb"] = round(free_mem, 2)
        gpu_info["vram_percent"] = pct
        gpu_info["device_capability"] = f"Compute {cap[0]}.{cap[1]}"

    # CPU Diagnostics
    cpu_cores_physical = psutil.cpu_count(logical=False) or 6
    cpu_cores_logical = psutil.cpu_count(logical=True) or 12
    cpu_percent = psutil.cpu_percent(interval=None)
    cpu_info = {
        "name": "11th Gen Intel(R) Core(TM) i5-11400H @ 2.70GHz",
        "cores": cpu_cores_physical,
        "threads": cpu_cores_logical,
        "usage_pct": cpu_percent,
        "architecture": platform.machine()
    }

    # RAM Diagnostics
    mem = psutil.virtual_memory()
    ram_info = {
        "total_gb": round(mem.total / (1024 ** 3), 2),
        "used_gb": round(mem.used / (1024 ** 3), 2),
        "free_gb": round(mem.available / (1024 ** 3), 2),
        "percent": mem.percent
    }

    # Count clips on disk
    disk_clips_count = 0
    for root, dirs, files in os.walk(CLIPS_DIR):
        if os.path.basename(root) == 'thumbnails':
            continue
        disk_clips_count += sum(1 for f in files if f.endswith('.mp4'))

    db_count = vector_store.count()
    unprocessed_count = max(0, disk_clips_count - db_count)

    return {
        "status": "online",
        "gpu": gpu_info,
        "cpu": cpu_info,
        "ram": ram_info,
        "vlm_model": {
            "name": "Qwen3-VL-4B-Instruct",
            "quantization": "4-bit NF4 (BitsAndBytes)",
            "attention": "Compiled SDPA / FlashAttention",
            "device": "NVIDIA GeForce RTX 2050 (Hybrid Overflow)",
            "context_tracking": "Chronological Incident Memory & Entity Re-ID Active"
        },
        "embedding_model": {
            "name": "Google VideoPrism (videoprism_lvt_public_v1_base)",
            "dimensions": 768,
            "context_window": 16,
            "device": "Host CPU / CUDA (0% VRAM contention)",
            "asymmetric": True
        },
        "vector_db": {
            "name": "ChromaDB (SQLite + HNSW)",
            "collection": vector_store.collection_name,
            "distance_metric": "Cosine Similarity",
            "total_indexed_clips": db_count,
            "total_disk_clips": disk_clips_count,
            "unprocessed_clips": unprocessed_count,
            "storage_path": vector_store.db_path
        }
    }


# ==============================================================================
# Real-Time Telemetry Loop & Startup Registration
# ==============================================================================
async def background_telemetry_loop():
    """Periodically broadcasts real-time telemetry (CPU, GPU, RAM, ChromaDB count) every 2 seconds."""
    while True:
        try:
            if ws_manager.has_listeners():
                status_data = await get_system_status()
                await ws_manager.broadcast({
                    "type": "system_telemetry",
                    "data": status_data
                })

                pipe_state = pipeline_instance.get_state()
                if pipe_state.get("status") in ["clipping", "analyzing"]:
                    await ws_manager.broadcast({
                        "type": "pipeline_state",
                        "data": pipe_state
                    })
        except Exception as e:
            pass
        await asyncio.sleep(2.0)


@app.on_event("startup")
async def startup_event():
    global main_loop
    main_loop = asyncio.get_running_loop()
    asyncio.create_task(background_telemetry_loop())
    print("[+] Main asyncio loop captured & background telemetry broadcast started.")


# ==============================================================================
# Server-Sent Events (SSE) Stream Endpoint
# ==============================================================================
@app.get("/api/events")
async def sse_events():
    """Server-Sent Events (SSE) real-time stream with automatic heartbeat."""
    queue = asyncio.Queue()
    await ws_manager.connect_sse(queue)

    async def sse_generator():
        try:
            init_status = await get_system_status()
            init_payload = {
                "type": "system_telemetry",
                "data": init_status
            }
            yield f"data: {json.dumps(init_payload)}\n\n"
            while True:
                msg = await queue.get()
                yield f"data: {json.dumps(msg)}\n\n"
        except (asyncio.CancelledError, GeneratorExit):
            pass
        finally:
            await ws_manager.disconnect_sse(queue)

    return StreamingResponse(
        sse_generator(),
        media_type="text/event-stream",
        headers={
            "Cache-Control": "no-cache",
            "Connection": "keep-alive",
            "X-Accel-Buffering": "no"
        }
    )


# ==============================================================================
# Video Library & Uploads
# ==============================================================================
@app.get("/api/videos")
async def list_videos():
    """Lists all available input videos in data/uploads and test/."""
    video_extensions = (".mp4", ".mkv", ".avi", ".mov")
    video_list = []

    def scan_dir(folder, category):
        if not os.path.exists(folder):
            return
        for fname in os.listdir(folder):
            if fname.lower().endswith(video_extensions):
                full_path = os.path.join(folder, fname)
                try:
                    size_mb = round(os.path.getsize(full_path) / (1024 * 1024), 2)
                    rel_url = f"/media/{category}/{fname}"
                    video_list.append({
                        "name": fname,
                        "path": full_path,
                        "category": category,
                        "size_mb": size_mb,
                        "url": rel_url
                    })
                except Exception:
                    pass

    scan_dir(UPLOADS_DIR, "uploads")
    scan_dir(TEST_DIR, "test")

    return {"videos": video_list}


@app.post("/api/upload")
async def upload_video(file: UploadFile = File(...)):
    """Uploads a new video file to data/uploads using chunked async streaming."""
    if not file.filename:
        raise HTTPException(status_code=400, detail="No file selected.")

    file_ext = os.path.splitext(file.filename)[1].lower()
    if file_ext not in [".mp4", ".mkv", ".avi", ".mov"]:
        raise HTTPException(status_code=400, detail="Invalid video format. Supported formats: MP4, MKV, AVI, MOV.")

    # Sanitize filename
    safe_name = os.path.basename(file.filename).replace(" ", "_")
    dest_path = os.path.join(UPLOADS_DIR, safe_name)

    try:
        # Stream 1MB chunks to disk
        CHUNK_SIZE = 1024 * 1024
        with open(dest_path, "wb") as buffer:
            while True:
                chunk = await file.read(CHUNK_SIZE)
                if not chunk:
                    break
                buffer.write(chunk)

        size_mb = round(os.path.getsize(dest_path) / (1024 * 1024), 2)
        print(f"[+] Successfully uploaded '{safe_name}' ({size_mb} MB) to {dest_path}")
        ws_manager.broadcast_threadsafe({"type": "videos_updated"})
        return {
            "status": "success",
            "filename": safe_name,
            "path": dest_path,
            "size_mb": size_mb,
            "url": f"/media/uploads/{safe_name}"
        }
    except Exception as e:
        if os.path.exists(dest_path):
            try:
                os.remove(dest_path)
            except Exception:
                pass
        raise HTTPException(status_code=500, detail=f"Upload error: {str(e)}")


# ==============================================================================
# Autonomous Pipeline Controls
# ==============================================================================
class PipelineStartRequest(BaseModel):
    video_path: str
    camera_id: str = "FRONT_PORCH_CAM_01"
    chunk_sec: float = 10.0
    resolution: str = "640x360"
    pixel_threshold: int = 18
    min_motion_percent: float = 0.5


@app.post("/api/pipeline/start")
async def start_pipeline(req: PipelineStartRequest):
    """Starts the full autonomous clipping + VLM + Nomic indexing pipeline."""
    state = pipeline_instance.get_state()
    if state["status"] in ["clipping", "analyzing"]:
        raise HTTPException(status_code=400, detail="Pipeline is already running.")

    if not os.path.exists(req.video_path):
        raise HTTPException(status_code=404, detail=f"Video file not found: {req.video_path}")

    # Launch background processing thread
    thread = threading.Thread(
        target=pipeline_instance.run_pipeline,
        kwargs={
            "video_path": req.video_path,
            "camera_id": req.camera_id,
            "chunk_sec": req.chunk_sec,
            "resolution": req.resolution,
            "pixel_threshold": req.pixel_threshold,
            "min_motion_percent": req.min_motion_percent,
            "status_callback": pipeline_callback
        },
        daemon=True
    )
    thread.start()
    ws_manager.broadcast_threadsafe({"type": "pipeline_state", "data": pipeline_instance.get_state()})
    ws_manager.broadcast_threadsafe({"type": "clips_updated"})

    return {"status": "started", "video": os.path.basename(req.video_path)}


@app.post("/api/pipeline/stop")
async def stop_pipeline():
    """Requests the autonomous pipeline to cancel/stop gracefully."""
    pipeline_instance.request_stop()
    ws_manager.broadcast_threadsafe({"type": "pipeline_state", "data": pipeline_instance.get_state()})
    return {"status": "stop_requested"}


@app.post("/api/pipeline/process-unprocessed")
async def process_unprocessed_pipeline():
    """Starts background processing of all unprocessed clips on disk with Qwen3-VL & ChromaDB."""
    state = pipeline_instance.get_state()
    if state["status"] in ["clipping", "analyzing"]:
        raise HTTPException(status_code=400, detail="Pipeline is already running.")

    thread = threading.Thread(
        target=pipeline_instance.process_unprocessed_clips,
        kwargs={"status_callback": pipeline_callback},
        daemon=True
    )
    thread.start()
    ws_manager.broadcast_threadsafe({"type": "pipeline_state", "data": pipeline_instance.get_state()})
    ws_manager.broadcast_threadsafe({"type": "clips_updated"})

    return {"status": "started", "message": "Batch processing of unprocessed clips started."}


@app.get("/api/pipeline/status")
async def get_pipeline_status():
    """Returns current live pipeline execution state."""
    return pipeline_instance.get_state()


# ==============================================================================
# All Clips Explorer & Semantic Vector Search (Processed + Unprocessed + 1-Day View)
# ==============================================================================
def parse_cctv_timestamp(raw_ts: Optional[str]):
    """
    Parses CCTV OCR timestamp strings like:
      - '28-10-2026 Wed 08:42:03'
      - '28-10-2026 Wed. 08:55:16'
      - '2026-10-28 08:42:03'
      - '28/10/2026 08:42:03'
    Returns: (date_iso_str, time_str, seconds_of_day)
      e.g. ('2026-10-28', '08:42:03', 31323)
    """
    if not raw_ts or not isinstance(raw_ts, str):
        return None, None, None
    clean_ts = raw_ts.strip()
    
    # 1. Extract Time: HH:MM:SS
    time_m = re.search(r"(\d{1,2}):(\d{2}):(\d{2})", clean_ts)
    t_str = None
    sec = None
    if time_m:
        h, m, s = int(time_m.group(1)), int(time_m.group(2)), int(time_m.group(3))
        t_str = f"{h:02d}:{m:02d}:{s:02d}"
        sec = h * 3600 + m * 60 + s

    # 2. Extract Date:
    d_str = None
    m_iso = re.search(r"(\d{4})[-/](\d{1,2})[-/](\d{1,2})", clean_ts)
    if m_iso:
        y, mo, d = int(m_iso.group(1)), int(m_iso.group(2)), int(m_iso.group(3))
        d_str = f"{y:04d}-{mo:02d}-{d:02d}"
    else:
        m_dmy = re.search(r"(\d{1,2})[-/](\d{1,2})[-/](\d{4})", clean_ts)
        if m_dmy:
            p1, p2, y = int(m_dmy.group(1)), int(m_dmy.group(2)), int(m_dmy.group(3))
            d, mo = (p1, p2) if p1 > 12 or p2 <= 12 else (p2, p1)
            d_str = f"{y:04d}-{mo:02d}-{d:02d}"

    return d_str, t_str, sec


@app.get("/api/clips")
async def get_clips(
    search: Optional[str] = Query(None, description="Natural language semantic search query"),
    sort_by: str = Query("ascending", description="Sort by: processed, unprocessed, ascending, descending, length, size, similarity"),
    order: str = Query("asc", description="Sort order: desc or asc"),
    date: Optional[str] = Query(None, description="Filter clips by specific date (YYYY-MM-DD)"),
    limit: int = Query(500, ge=1, le=1000)
):
    """
    Retrieves all clips (both processed in ChromaDB and unprocessed on disk).
    Filters to a single day by footage timestamp by default or as requested via date.
    Supports sorting by CCTV footage timestamp: [processed, unprocessed, ascending, descending, length, size, similarity].
    """
    # Normalize parameters (support both FastAPI dependency injection and direct python calls)
    clean_search = search.strip() if (search and isinstance(search, str) and search.strip()) else None
    sort_key = sort_by.lower() if isinstance(sort_by, str) else "ascending"
    filter_date = date if (date and isinstance(date, str)) else None
    clean_limit = int(limit) if isinstance(limit, (int, float)) else 500

    # 1. Fetch ChromaDB items (or run semantic search if search query provided)
    db_items = {}
    if clean_search:
        search_res = vector_store.search_clips(query=clean_search, n_results=clean_limit)
        for r in search_res:
            db_items[r["clip_id"]] = {
                "metadata": r.get("metadata", {}),
                "document": r.get("document", ""),
                "similarity": r.get("similarity"),
                "confidence_pct": r.get("confidence_pct")
            }
    else:
        raw_db = vector_store._collection.get(include=["metadatas", "documents"])
        if raw_db and raw_db["ids"]:
            for i, cid in enumerate(raw_db["ids"]):
                db_items[cid] = {
                    "metadata": raw_db["metadatas"][i] if raw_db["metadatas"] else {},
                    "document": raw_db["documents"][i] if raw_db["documents"] else "",
                    "similarity": None,
                    "confidence_pct": None
                }

    # 2. Pre-scan video folders to determine footage consensus dates and base seconds
    video_dates: Dict[str, List[str]] = {}
    video_base_secs: Dict[str, List[float]] = {}

    for root, dirs, files in os.walk(CLIPS_DIR):
        if os.path.basename(root) == "thumbnails":
            continue
        v_base = os.path.basename(root)
        if v_base not in video_dates:
            video_dates[v_base] = []
            video_base_secs[v_base] = []

        for f in files:
            if not f.endswith(".mp4"):
                continue
            cid = os.path.splitext(f)[0]
            if cid in db_items:
                meta = db_items[cid]["metadata"]
                raw_ts = meta.get("timestamp_str") or ""
                d_str, t_str, s_day = parse_cctv_timestamp(raw_ts)
                if d_str:
                    video_dates[v_base].append(d_str)
                if s_day is not None:
                    # Calculate base second at t=0 of footage
                    s_sec = float(meta.get("start_sec", 0.0))
                    if s_sec == 0.0:
                        m = re.match(r"event_clip_(\d+)_(\d+)m(\d+)s_to_(\d+)m(\d+)s", cid)
                        if m:
                            _, m1, s1, _, _ = map(int, m.groups())
                            s_sec = float(m1 * 60 + s1)
                    video_base_secs[v_base].append(s_day - s_sec)

    consensus_dates: Dict[str, str] = {}
    consensus_base_secs: Dict[str, float] = {}
    for v_base in video_dates:
        if video_dates[v_base]:
            consensus_dates[v_base] = Counter(video_dates[v_base]).most_common(1)[0][0]
        if video_base_secs[v_base]:
            consensus_base_secs[v_base] = statistics.median(video_base_secs[v_base])

    # 3. Scan all MP4 clips on disk in data/clips/
    all_clips = []
    for root, dirs, files in os.walk(CLIPS_DIR):
        if os.path.basename(root) == "thumbnails":
            continue
        video_base = os.path.basename(root)
        video_c_date = consensus_dates.get(video_base)
        video_c_base = consensus_base_secs.get(video_base)

        for f in files:
            if not f.endswith(".mp4"):
                continue

            clip_id = os.path.splitext(f)[0]
            mp4_path = os.path.join(root, f)
            size_mb = round(os.path.getsize(mp4_path) / (1024 * 1024), 2)
            mtime = os.path.getmtime(mp4_path)
            cdate = time.strftime("%Y-%m-%d", time.localtime(mtime))
            time_display = time.strftime("%Y-%m-%d %H:%M:%S", time.localtime(mtime))

            # Parse start_sec, end_sec, duration from filename
            start_sec = 0.0
            dur_sec = 0.0
            placeholder_title = f"Event ({clip_id})"
            time_range_str = ""

            m = re.match(r"event_clip_(\d+)_(\d+)m(\d+)s_to_(\d+)m(\d+)s", clip_id)
            if m:
                idx, m1, s1, m2, s2 = map(int, m.groups())
                start_sec = float(m1 * 60 + s1)
                end_sec = float(m2 * 60 + s2)
                dur_sec = round(max(0.0, end_sec - start_sec), 1)
                time_range_str = f"{m1:02d}:{s1:02d} - {m2:02d}:{s2:02d}"
                placeholder_title = f"Motion Event #{idx:02d} ({time_range_str})"

            video_url = f"/media/clips/{video_base}/{f}"
            thumb_url = f"/media/clips/{video_base}/thumbnails/{clip_id}.jpg"

            if clip_id in db_items:
                # Processed clip with forensic report
                meta = db_items[clip_id]["metadata"]
                doc = db_items[clip_id]["document"]
                sim = db_items[clip_id]["similarity"]

                raw_ts = meta.get("timestamp_str") or ""
                d_str, t_str, sec_of_day = parse_cctv_timestamp(raw_ts)

                clip_date = d_str or video_c_date or cdate

                if sec_of_day is not None:
                    sort_sec = float(sec_of_day)
                elif video_c_base is not None:
                    sort_sec = float(video_c_base + float(meta.get("start_sec", start_sec)))
                else:
                    sort_sec = float(meta.get("start_sec", start_sec))

                actual_ts = raw_ts if raw_ts else time_display
                if not actual_ts or actual_ts == time_display:
                    if video_c_date and video_c_base is not None:
                        cs = int(sort_sec) % 86400
                        actual_ts = f"{video_c_date} {cs // 3600:02d}:{(cs % 3600) // 60:02d}:{cs % 60:02d}"

                all_clips.append({
                    "clip_id": clip_id,
                    "status": "processed",
                    "title": meta.get("title") or placeholder_title,
                    "caption": meta.get("direction_of_motion") or "Analyzed Motion Trajectory",
                    "timestamp": actual_ts,
                    "action_summary": meta.get("action_summary") or "",
                    "camera_id": meta.get("camera_id") or "FRONT_PORCH_CAM_01",
                    "start_sec": float(meta.get("start_sec", start_sec)),
                    "duration_sec": float(meta.get("duration_sec", dur_sec)),
                    "size_mb": size_mb,
                    "date": clip_date,
                    "sort_seconds": sort_sec,
                    "time_range": time_range_str,
                    "video_url": meta.get("video_url") or video_url,
                    "thumbnail_url": meta.get("thumbnail_url") or thumb_url,
                    "full_report": doc,
                    "similarity": sim,
                    "confidence_pct": db_items[clip_id].get("confidence_pct")
                })
            else:
                # Unprocessed clip on disk (awaiting VLM analysis)
                clip_date = video_c_date or cdate
                if video_c_base is not None:
                    sort_sec = float(video_c_base + start_sec)
                    cs = int(sort_sec) % 86400
                    unprocessed_ts = f"{clip_date} {cs // 3600:02d}:{(cs % 3600) // 60:02d}:{cs % 60:02d}"
                else:
                    sort_sec = float(start_sec)
                    unprocessed_ts = time_display

                all_clips.append({
                    "clip_id": clip_id,
                    "status": "unprocessed",
                    "title": placeholder_title,
                    "caption": "Unprocessed Event (Awaiting VLM)",
                    "timestamp": unprocessed_ts,
                    "action_summary": "Extracted motion event clip. Awaiting Qwen3-VL hierarchical forensic synthesis.",
                    "camera_id": "FRONT_PORCH_CAM_01",
                    "start_sec": start_sec,
                    "duration_sec": dur_sec,
                    "size_mb": size_mb,
                    "date": clip_date,
                    "sort_seconds": sort_sec,
                    "time_range": time_range_str,
                    "video_url": video_url,
                    "thumbnail_url": thumb_url,
                    "full_report": "",
                    "similarity": None,
                    "confidence_pct": None
                })

    # 4. Available Dates calculation (sorted descending)
    available_dates = sorted(list(set(c["date"] for c in all_clips)), reverse=True)
    if not available_dates:
        available_dates = [time.strftime("%Y-%m-%d")]

    # 5. 1-Day View Filter: Filter by selected date or default to the most recent date with clips
    selected_date = filter_date if (filter_date and filter_date in available_dates) else available_dates[0]
    filtered_clips = [c for c in all_clips if c["date"] == selected_date]

    # 6. Sorting implementation for: [processed, unprocessed, ascending, descending, length, size]
    if clean_search:
        # Relevance search prioritizes high similarity matches first, then chronological
        filtered_clips.sort(
            key=lambda c: (
                1 if c["status"] == "processed" else 0,
                c["similarity"] if c["similarity"] is not None else -1.0,
                -c["sort_seconds"]
            ),
            reverse=True
        )
    elif sort_key == "processed":
        # Processed first, then by timeline ascending
        filtered_clips.sort(key=lambda c: (0 if c["status"] == "processed" else 1, c["sort_seconds"]))
    elif sort_key == "unprocessed":
        # Unprocessed first, then by timeline ascending
        filtered_clips.sort(key=lambda c: (0 if c["status"] == "unprocessed" else 1, c["sort_seconds"]))
    elif sort_key in ["ascending", "timeline"]:
        # Earliest in the day first: 00:00 -> 23:59
        filtered_clips.sort(key=lambda c: c["sort_seconds"])
    elif sort_key == "descending":
        # Latest in the day first: 23:59 -> 00:00
        filtered_clips.sort(key=lambda c: c["sort_seconds"], reverse=True)
    elif sort_key == "length":
        # Longest duration first, then chronological
        filtered_clips.sort(key=lambda c: (c["duration_sec"], -c["sort_seconds"]), reverse=True)
    elif sort_key == "size":
        # Largest file size first, then chronological
        filtered_clips.sort(key=lambda c: (c["size_mb"], -c["sort_seconds"]), reverse=True)
    else:
        filtered_clips.sort(key=lambda c: c["sort_seconds"])

    # Counts
    count_processed = sum(1 for c in filtered_clips if c["status"] == "processed")
    count_unprocessed = sum(1 for c in filtered_clips if c["status"] == "unprocessed")

    return {
        "selected_date": selected_date,
        "available_dates": available_dates,
        "count": len(filtered_clips),
        "total_processed": count_processed,
        "total_unprocessed": count_unprocessed,
        "total_all_days": len(all_clips),
        "clips": filtered_clips
    }


@app.post("/api/clips/{clip_id}/analyze")
async def analyze_single_clip(clip_id: str):
    """Analyzes a single unprocessed clip on demand using Qwen3-VL and indexes it into ChromaDB."""
    # Find clip on disk
    target_mp4 = None
    target_video_base = None
    for root, dirs, files in os.walk(CLIPS_DIR):
        if os.path.basename(root) == "thumbnails":
            continue
        fname = f"{clip_id}.mp4"
        if fname in files:
            target_mp4 = os.path.join(root, fname)
            target_video_base = os.path.basename(root)
            break

    if not target_mp4 or not os.path.exists(target_mp4):
        raise HTTPException(status_code=404, detail=f"Clip '{clip_id}' not found on disk.")

    # Lazy-load Qwen3-VL
    pipeline_instance._ensure_vlm_loaded()
    parsed_res = (640, 360)

    # Extract timeline start second & end second
    target_start_sec = 0.0
    target_end_sec = 0.0
    m_ts = re.match(r"event_clip_(\d+)_(\d+)m(\d+)s_to_(\d+)m(\d+)s", clip_id)
    if m_ts:
        _, m1, s1, m2, s2 = map(int, m_ts.groups())
        target_start_sec = float(m1 * 60 + s1)
        target_end_sec = float(m2 * 60 + s2)
    target_duration_sec = max(0.0, target_end_sec - target_start_sec)

    try:
        # Visual RAG Step 1: Pre-encode video into 768-d Google VideoPrism embedding
        clip_emb = vector_store.embedder.embed_video(target_mp4)

        # Visual RAG Step 2: Query ChromaDB for prior events with high visual similarity
        rag_matches = vector_store.retrieve_prior_context_rag(
            clip_embedding=clip_emb,
            current_start_sec=target_start_sec,
            camera_id="FRONT_PORCH_CAM_01",
            top_k=3,
            min_cosine_threshold=0.88
        )
        prior_context = vector_store.format_rag_context_for_prompt(rag_matches)
        if rag_matches:
            print(f"[*] [Visual RAG] Retrieved {len(rag_matches)} prior match(es) for {clip_id}:")
            for rm in rag_matches:
                print(f"    -> {rm['clip_id']} (similarity={rm['cosine_similarity']:.4f})")
        else:
            print(f"[*] [Visual RAG] No previous matches above threshold for {clip_id} (novel scene/entity).")

        # Visual RAG Step 3: Run VLM Hierarchical Analysis with retrieved context
        from test_qwen3_pipeline import analyze_video_hierarchical
        forensic_report = analyze_video_hierarchical(
            model=pipeline_instance._vlm_model,
            processor=pipeline_instance._vlm_processor,
            video_source=target_mp4,
            chunk_duration_sec=10.0,
            target_size=parsed_res,
            max_new_tokens=4096,
            prior_context=prior_context
        )

        parsed = vector_store.parse_forensic_report(forensic_report)
        thumb_url = f"/media/clips/{target_video_base}/thumbnails/{clip_id}.jpg"
        video_url = f"/media/clips/{target_video_base}/{clip_id}.mp4"

        # Visual RAG Step 4: Index into ChromaDB reusing precomputed clip_emb (0 duplicate compute)
        vector_store.add_clip(
            clip_id=clip_id,
            forensic_report=forensic_report,
            video_path=target_mp4,
            embedding=clip_emb,
            metadata={
                "camera_id": "FRONT_PORCH_CAM_01",
                "clip_path": target_mp4,
                "video_url": video_url,
                "thumbnail_url": thumb_url,
                "start_sec": target_start_sec,
                "end_sec": target_end_sec,
                "duration_sec": target_duration_sec,
                "source_video": target_video_base
            }
        )

        ws_manager.broadcast_threadsafe({"type": "clips_updated", "clip_id": clip_id})

        return {
            "status": "success",
            "clip_id": clip_id,
            "title": parsed["title"],
            "caption": parsed["direction_of_motion"],
            "action_summary": parsed["action_summary"],
            "forensic_report": forensic_report
        }
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Analysis failed: {str(e)}")


@app.delete("/api/clips/{clip_id}")
async def delete_clip(clip_id: str):
    """Deletes a clip from ChromaDB."""
    success = vector_store.delete_clip(clip_id)
    if not success:
        raise HTTPException(status_code=404, detail="Clip not found or delete failed.")
    ws_manager.broadcast_threadsafe({"type": "clips_updated", "clip_id": clip_id})
    return {"status": "deleted", "clip_id": clip_id, "remaining": vector_store.count()}


@app.post("/api/clips/reset")
async def reset_database():
    """Wipes all indexed clips in ChromaDB."""
    vector_store.reset_db()
    ws_manager.broadcast_threadsafe({"type": "clips_updated"})
    return {"status": "reset", "count": 0}


# ==============================================================================
# WebSocket Endpoint
# ==============================================================================
@app.websocket("/ws/pipeline")
async def websocket_pipeline(websocket: WebSocket):
    await ws_manager.connect(websocket)
    try:
        # Send initial pipeline state & telemetry immediately
        await websocket.send_text(json.dumps({
            "type": "pipeline_state",
            "data": pipeline_instance.get_state()
        }))
        init_sys = await get_system_status()
        await websocket.send_text(json.dumps({
            "type": "system_telemetry",
            "data": init_sys
        }))
        while True:
            # Keepalive / ping
            await websocket.receive_text()
    except WebSocketDisconnect:
        await ws_manager.disconnect(websocket)
    except Exception:
        await ws_manager.disconnect(websocket)


# ==============================================================================
# Serve Production Built React Frontend if Available
# ==============================================================================
FRONTEND_DIST = os.path.join(BASE_DIR, "frontend", "dist")
if os.path.exists(FRONTEND_DIST):
    app.mount("/", StaticFiles(directory=FRONTEND_DIST, html=True), name="frontend")


# ==============================================================================
# Run with python server.py
# ==============================================================================
if __name__ == "__main__":
    import uvicorn
    print("=" * 70)
    print("[*] STARTING NEURAL CCTV FASTAPI SERVER (http://127.0.0.1:8000)")
    print("=" * 70)
    uvicorn.run(app, host="127.0.0.1", port=8000)
