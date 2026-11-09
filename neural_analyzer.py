"""
================================================================================
Neural CCTV Analyser - Smart Household Video Analytics Engine
================================================================================
Capabilities:
  1. Anchor-based temporal motion gating (10-frame chunks) saving 90%+ GPU idle load
  2. Household polygonal virtual fencing (Doorstep, Driveway, Porch Walkway)
  3. YOLOv8 object & package/parcel tracking
  4. Local 512-d Face Vector Recognition with ChromaDB (Recognizes Father, Mother, etc.)
  5. Automatic unknown face snapshot capture for user labeling
  6. High-level semantic natural language activity logging
  7. Optional annotated video export with security HUD overlays
"""

import os
import sys
import json
import time
import argparse
from datetime import datetime, timedelta
from typing import Dict, List, Tuple, Optional, Any

import cv2
import numpy as np
import torch
from ultralytics import YOLO

from neural_cctv.motion_gate import MotionGate
from neural_cctv.face_engine import FaceVectorEngine
from neural_cctv.semantic_engine import SemanticActivityEngine, HouseholdZone
from neural_cctv.osd_extractor import OSDTimestampExtractor


class NeuralCCTVEngine:
    """
    Main orchestration engine for Neural CCTV Analyser.
    """

    @staticmethod
    def _detect_carried_package(person_crop: np.ndarray) -> bool:
        """
        Inspects the torso / mid-body area for carried packages, boxes, or parcels
        even when standard COCO classes don't explicitly classify custom courier boxes.
        """
        if person_crop is None or person_crop.size == 0:
            return False
        h, w = person_crop.shape[:2]
        if h < 60 or w < 30:
            return False
        # Mid-body torso region where packages are carried
        torso = person_crop[int(h * 0.28):int(h * 0.72), int(w * 0.15):int(w * 0.85)]
        if torso.size == 0:
            return False
        gray = cv2.cvtColor(torso, cv2.COLOR_BGR2GRAY)
        edges = cv2.Canny(gray, 40, 130)
        contours, _ = cv2.findContours(edges, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        torso_area = float(torso.shape[0] * torso.shape[1])
        for c in contours:
            area = cv2.contourArea(c)
            if area > (torso_area * 0.12):
                x, y, cw, ch = cv2.boundingRect(c)
                aspect = cw / float(max(ch, 1))
                if 0.45 < aspect < 2.8:
                    return True
        return False

    COCO_PACKAGE_CLASSES = {24: "backpack", 26: "handbag", 28: "suitcase"}
    COCO_VEHICLE_CLASSES = {2: "car", 3: "motorcycle", 5: "bus", 7: "truck"}

    def __init__(self, config_path: str = "config.json", device: Optional[str] = None):
        self.config_path = config_path
        with open(config_path, "r", encoding="utf-8") as f:
            self.config = json.load(f)

        # Device selection
        if device and device != "auto":
            self.device = device
        else:
            self.device = "cuda" if torch.cuda.is_available() else "cpu"

        print(f"[Neural CCTV] Initializing engine on device: {self.device.upper()} ...")

        # 1. Motion Gater
        mg_cfg = self.config.get("motion_gate", {})
        self.chunk_size = int(mg_cfg.get("chunk_size", 10))
        self.motion_threshold = float(mg_cfg.get("motion_threshold", 3.5))
        self.motion_gate = MotionGate(
            chunk_size=self.chunk_size,
            motion_threshold=self.motion_threshold,
            downsample_size=tuple(mg_cfg.get("downsample_size", [320, 180]))
        )

        # 2. Face Recognition & Local ChromaDB
        fc_cfg = self.config.get("face_recognition", {})
        self.face_engine = FaceVectorEngine(
            db_path=fc_cfg.get("db_path", "./data/face_db"),
            unknown_dir=fc_cfg.get("unknown_dir", "./data/unknown_faces"),
            distance_threshold=float(fc_cfg.get("distance_threshold", 0.38)),
            device=self.device
        )

        # 3. Semantic Activity Engine & Zones
        zones_cfg = self.config.get("zones", [])
        log_cfg = self.config.get("logging", {})
        self.semantic_engine = SemanticActivityEngine(
            zones_config=zones_cfg,
            log_cooldown_seconds=float(log_cfg.get("cooldown_seconds", 3.5))
        )

        # 4. YOLO Object Detector
        yolo_cfg = self.config.get("yolo", {})
        model_path = yolo_cfg.get("model_path", "yolov8n.pt")
        self.yolo_model = YOLO(model_path)
        self.yolo_conf = float(yolo_cfg.get("confidence", 0.35))

        # 5. Virtual Fence line
        vf_cfg = self.config.get("virtual_fence", {})
        self.virtual_fence_enabled = vf_cfg.get("enabled", True)
        self.virtual_fence_line = vf_cfg.get("line", [[0.05, 0.65], [0.95, 0.65]])

        # 6. Top-Left OSD Timestamp Extractor
        self.osd_extractor = OSDTimestampExtractor()

        # Statistics
        self.stats = {
            "total_frames": 0,
            "total_chunks": 0,
            "skipped_chunks": 0,
            "active_chunks": 0,
            "events_logged": 0,
            "known_faces_recognized": 0,
            "unknown_faces_captured": 0
        }

    def format_timecode(self, seconds: float) -> str:
        td = timedelta(seconds=seconds)
        total_seconds = int(td.total_seconds())
        hours = total_seconds // 3600
        minutes = (total_seconds % 3600) // 60
        secs = total_seconds % 60
        millis = int((seconds - int(seconds)) * 1000)
        return f"{hours:02d}:{minutes:02d}:{secs:02d}.{millis:03d}"

    def process_video(
        self,
        video_path: str,
        output_log_path: str = "cctv_activity.log",
        json_log_path: Optional[str] = "cctv_activity.jsonl",
        save_video_path: Optional[str] = None,
        display: bool = False
    ):
        cap = cv2.VideoCapture(video_path)
        if not cap.isOpened():
            raise FileNotFoundError(f"Could not open video source: {video_path}")

        fps = cap.get(cv2.CAP_PROP_FPS) or 30.0
        width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
        height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
        total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))

        print(f"[Neural CCTV] Ingesting: {video_path} ({width}x{height} @ {fps:.1f} FPS, {total_frames} frames)")
        print(f"[Neural CCTV] 10-Frame Motion Gate Active (Threshold: {self.motion_threshold} px)")
        print(f"[Neural CCTV] Logging activity to: {output_log_path}")

        # Check for OSD timestamp in top-left corner
        ret_init, first_frame = cap.read()
        cap.set(cv2.CAP_PROP_POS_FRAMES, 0)
        use_osd = False
        if ret_init:
            osd_init = self.osd_extractor.extract(first_frame, 0, fps, force_ocr=True)
            if not osd_init.startswith("Frame"):
                use_osd = True
                print(f"[Neural CCTV] Top-Left OSD Timestamp Detected: '{osd_init}'")

        # Setup logs
        os.makedirs(os.path.dirname(os.path.abspath(output_log_path)) or ".", exist_ok=True)
        log_f = open(output_log_path, "w", encoding="utf-8")
        json_f = open(json_log_path, "w", encoding="utf-8") if json_log_path else None

        log_ts_label = "OSD TIMESTAMP" if use_osd else "TIMECODE"
        log_f.write("=" * 115 + "\n")
        log_f.write(" " * 32 + "NEURAL CCTV ANALYSER - HOUSEHOLD SURVEILLANCE LOG\n")
        log_f.write("=" * 115 + "\n")
        log_f.write(f"Session Start: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}\n")
        log_f.write(f"Camera ID    : {self.config.get('camera', {}).get('camera_id', 'FRONT_PORCH')}\n")
        log_f.write("-" * 115 + "\n")
        log_f.write(f"{log_ts_label:<22} | {'FRAME':<7} | {'SEVERITY':<8} | {'CATEGORY':<24} | DESCRIPTION\n")
        log_f.write("-" * 115 + "\n")
        log_f.flush()

        # Video Writer if save_video is set
        writer = None
        if save_video_path:
            fourcc = cv2.VideoWriter_fourcc(*'mp4v')
            writer = cv2.VideoWriter(save_video_path, fourcc, fps, (width, height))
            print(f"[Neural CCTV] Saving annotated visualization to: {save_video_path}")

        chunk_frames = []
        chunk_indices = []
        frame_idx = 0
        latest_banner = "System Monitoring: Scene Idle"

        try:
            while True:
                ret, frame = cap.read()
                if not ret:
                    break

                chunk_frames.append(frame)
                chunk_indices.append(frame_idx)
                frame_idx += 1
                self.stats["total_frames"] += 1

                # Once we accumulate a chunk (e.g. 10 frames), evaluate motion gate
                if len(chunk_frames) >= self.chunk_size:
                    self.stats["total_chunks"] += 1
                    is_active, mean_diff, max_diff, peak_idx, peak_frame = self.motion_gate.evaluate_chunk(
                        chunk_frames, chunk_indices
                    )

                    chunk_time_sec = chunk_indices[0] / fps
                    timecode = self.format_timecode(chunk_time_sec)

                    if frame_idx % 1500 < self.chunk_size:
                        pct = (frame_idx / max(total_frames, 1)) * 100.0
                        cur_clock = self.osd_extractor.extract(chunk_frames[0], frame_idx, fps) if use_osd else timecode
                        tot = max(self.stats["total_chunks"], 1)
                        skp = self.stats["skipped_chunks"]
                        print(f"[Neural CCTV] Progress: Frame {frame_idx}/{total_frames} ({pct:.1f}%) | OSD Time: {cur_clock} | Idle: {(skp/tot)*100.0:.1f}% skipped | Events: {self.stats['events_logged']}")

                    if not is_active:
                        # -------------------------------------------------------------
                        # IDLE CHUNK: SKIP GPU PIPELINE ENTIRELY!
                        # -------------------------------------------------------------
                        self.stats["skipped_chunks"] += 1
                        latest_banner = f"Motion Gate: IDLE ({mean_diff:.1f}px delta) - GPU Sleeping"

                        for f in chunk_frames:
                            if writer or display:
                                annotated = self._draw_overlay(f, width, height, timecode, latest_banner, is_idle=True)
                                if writer:
                                    writer.write(annotated)
                                if display:
                                    cv2.imshow("Neural CCTV Analyser", annotated)
                                    if cv2.waitKey(1) & 0xFF == ord('q'):
                                        raise KeyboardInterrupt
                    else:
                        # -------------------------------------------------------------
                        # ACTIVE CHUNK: RUN NEURAL DETECTION, FACE EMBEDDINGS & SEMANTICS
                        # -------------------------------------------------------------
                        self.stats["active_chunks"] += 1

                        for f_idx, current_frame in zip(chunk_indices, chunk_frames):
                            # Stride 2 on CPU active chunks: process alternate frames while keeping peak frame
                            if (f_idx % 2 != 0) and (f_idx != peak_idx):
                                continue
                            cur_time_sec = f_idx / fps
                            cur_timecode = self.format_timecode(cur_time_sec)

                            # Run YOLOv8 tracking
                            results = self.yolo_model.track(
                                current_frame,
                                persist=True,
                                verbose=False,
                                conf=self.yolo_conf,
                                device=self.device
                            )

                            detected_packages = []
                            detected_persons = []
                            detected_vehicles = []

                            if results and len(results) > 0 and results[0].boxes is not None:
                                boxes = results[0].boxes
                                for box in boxes:
                                    cls_id = int(box.cls[0])
                                    x1, y1, x2, y2 = map(int, box.xyxy[0].tolist())
                                    conf = float(box.conf[0])
                                    track_id = int(box.id[0]) if box.id is not None else None

                                    # Check packages / parcels
                                    if cls_id in self.COCO_PACKAGE_CLASSES:
                                        detected_packages.append([x1, y1, x2, y2])

                                    # Check persons
                                    elif cls_id == 0:  # COCO class 0: Person
                                        detected_persons.append((track_id, [x1, y1, x2, y2], conf))

                                    # Check vehicles
                                    elif cls_id in self.COCO_VEHICLE_CLASSES:
                                        v_type = self.COCO_VEHICLE_CLASSES[cls_id]
                                        detected_vehicles.append((track_id, v_type, [x1, y1, x2, y2], conf))

                            # Update vehicle tracks in semantic engine
                            for t_id, v_type, v_box, conf in detected_vehicles:
                                if t_id is not None:
                                    self.semantic_engine.update_track(
                                        track_id=t_id,
                                        entity_type=v_type,
                                        bbox=v_box,
                                        frame_w=width,
                                        frame_h=height,
                                        timestamp=cur_time_sec
                                    )

                            # Update person tracks and run Face Recognition
                            for t_id, p_box, conf in detected_persons:
                                if t_id is None:
                                    continue

                                px1, py1, px2, py2 = p_box
                                person_crop = current_frame[max(0, py1):py2, max(0, px1):px2]
                                has_pkg = self._detect_carried_package(person_crop)

                                track = self.semantic_engine.update_track(
                                    track_id=t_id,
                                    entity_type="person",
                                    bbox=p_box,
                                    frame_w=width,
                                    frame_h=height,
                                    timestamp=cur_time_sec,
                                    detected_packages=detected_packages
                                )
                                if has_pkg:
                                    track.carrying_package = True

                                # Attempt face identification if not yet identified
                                if not track.is_identified:
                                    px1, py1, px2, py2 = p_box
                                    # Face region is typically the top 35% of the person crop
                                    face_h = int((py2 - py1) * 0.38)
                                    face_crop = current_frame[max(0, py1):py1 + face_h, max(0, px1):px2]

                                    if face_crop.shape[0] > 30 and face_crop.shape[1] > 30:
                                        is_known, name, dist, emb = self.face_engine.identify_face(
                                            face_crop,
                                            track_id=t_id,
                                            timestamp_str=datetime.now().strftime("%Y%m%d_%H%M%S")
                                        )
                                        if is_known:
                                            self.semantic_engine.set_identity(t_id, name, True, 1.0 - dist)
                                            self.stats["known_faces_recognized"] += 1
                                        else:
                                            self.stats["unknown_faces_captured"] += 1

                            # Generate semantic natural language events
                            events = self.semantic_engine.generate_events(cur_time_sec)
                            for ev in events:
                                self.stats["events_logged"] += 1
                                latest_banner = ev["description"]
                                cur_stamp = self.osd_extractor.extract(current_frame, f_idx, fps) if use_osd else cur_timecode
                                log_line = f"{cur_stamp:<22} | {f_idx:<7} | {ev['severity']:<8} | {ev['category']:<24} | {ev['description']}\n"
                                log_f.write(log_line)
                                log_f.flush()

                                # High-contrast console notification
                                color = "\033[92m" if ev["severity"] == "INFO" else ("\033[93m" if ev["severity"] == "NOTICE" else "\033[91m")
                                reset = "\033[0m"
                                print(f"{color}[{cur_stamp}] [{ev['severity']}] {ev['description']}{reset}")

                                if json_f:
                                    json_record = {
                                        "timestamp": cur_stamp,
                                        "frame": f_idx,
                                        "severity": ev["severity"],
                                        "category": ev["category"],
                                        "track_id": ev["track_id"],
                                        "identity": ev["identity"],
                                        "zone": ev["zone"],
                                        "description": ev["description"]
                                    }
                                    json_f.write(json.dumps(json_record) + "\n")
                                    json_f.flush()

                            # Draw annotation
                            if writer or display:
                                annotated = self._draw_overlay(
                                    current_frame, width, height, cur_timecode, latest_banner, is_idle=False
                                )
                                if writer:
                                    writer.write(annotated)
                                if display:
                                    cv2.imshow("Neural CCTV Analyser", annotated)
                                    if cv2.waitKey(1) & 0xFF == ord('q'):
                                        raise KeyboardInterrupt

                    chunk_frames = []
                    chunk_indices = []

        except KeyboardInterrupt:
            print("\n[Neural CCTV] Processing stopped by user.")
        finally:
            cap.release()
            if writer:
                writer.release()
            if display:
                cv2.destroyAllWindows()
            log_f.close()
            if json_f:
                json_f.close()

        self._print_session_summary()

    def _draw_overlay(
        self,
        frame: np.ndarray,
        w: int,
        h: int,
        timecode: str,
        banner_text: str,
        is_idle: bool = False
    ) -> np.ndarray:
        """Renders virtual fences, household zones, and semantic activity banner."""
        canvas = frame.copy()

        # 1. Draw Household Polygonal Zones
        for zone in self.semantic_engine.zones:
            pts = np.array([[int(px * w), int(py * h)] for px, py in zone.polygon_norm], dtype=np.int32)
            color = (0, 165, 255) if "doorstep" in zone.name.lower() else (255, 144, 30)
            overlay = canvas.copy()
            cv2.fillPoly(overlay, [pts], color)
            cv2.addWeighted(overlay, 0.15, canvas, 0.85, 0, canvas)
            cv2.polylines(canvas, [pts], True, color, 2)
            cx, cy = int(np.mean(pts[:, 0])), int(np.mean(pts[:, 1]))
            cv2.putText(canvas, zone.name.upper(), (cx - 40, cy),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 1, cv2.LINE_AA)

        # 2. Draw Virtual Fence Line
        if self.virtual_fence_enabled and self.virtual_fence_line:
            p1 = (int(self.virtual_fence_line[0][0] * w), int(self.virtual_fence_line[0][1] * h))
            p2 = (int(self.virtual_fence_line[1][0] * w), int(self.virtual_fence_line[1][1] * h))
            cv2.line(canvas, p1, p2, (0, 255, 255), 2, cv2.LINE_AA)
            cv2.putText(canvas, "VIRTUAL FENCE", (p1[0] + 15, p1[1] - 8),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.45, (0, 255, 255), 1, cv2.LINE_AA)

        # 3. Draw Active Track Bounding Boxes
        if not is_idle:
            for t_id, track in self.semantic_engine.tracks.items():
                if track.bboxes:
                    bx1, by1, bx2, by2 = track.bboxes[-1]
                    color = (0, 255, 0) if track.is_identified else (0, 140, 255)
                    cv2.rectangle(canvas, (bx1, by1), (bx2, by2), color, 2)

                    pkg_str = " + [Package]" if track.carrying_package else ""
                    label = f"{track.identity_name} #{t_id}{pkg_str}"
                    cv2.rectangle(canvas, (bx1, max(0, by1 - 22)), (bx1 + len(label) * 9 + 12, by1), color, -1)
                    cv2.putText(canvas, label, (bx1 + 4, max(14, by1 - 6)),
                                cv2.FONT_HERSHEY_SIMPLEX, 0.48, (0, 0, 0), 1, cv2.LINE_AA)

        # 4. Top Status Banner (Activity Description)
        banner_bg_color = (30, 30, 30) if is_idle else (15, 60, 20)
        cv2.rectangle(canvas, (0, 0), (w, 42), banner_bg_color, -1)
        cv2.line(canvas, (0, 42), (w, 42), (100, 100, 100), 1)

        cv2.putText(canvas, f"[{timecode}]", (12, 26),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.55, (200, 200, 200), 1, cv2.LINE_AA)
        cv2.putText(canvas, banner_text, (140, 26),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.58, (0, 255, 255) if not is_idle else (180, 180, 180), 2, cv2.LINE_AA)

        return canvas

    def _print_session_summary(self):
        total_chunks = max(self.stats["total_chunks"], 1)
        savings_pct = (self.stats["skipped_chunks"] / total_chunks) * 100.0

        print("\n" + "=" * 65)
        print("          NEURAL CCTV ANALYSER - SESSION SUMMARY")
        print("=" * 65)
        print(f" Total Video Frames Processed : {self.stats['total_frames']}")
        print(f" 10-Frame Motion Chunks       : {self.stats['total_chunks']}")
        print(f" Motion Gate Skipped Chunks   : {self.stats['skipped_chunks']} ({savings_pct:.1f}% GPU Compute Saved!)")
        print(f" Active Neural Chunks Evaluated: {self.stats['active_chunks']}")
        print(f" Semantic Events Logged       : {self.stats['events_logged']}")
        print(f" Known Faces Identified       : {self.stats['known_faces_recognized']}")
        print(f" Unknown Faces Captured       : {self.stats['unknown_faces_captured']}")
        print("=" * 65 + "\n")


def main():
    parser = argparse.ArgumentParser(description="Neural CCTV Analyser - Smart Household Video Analytics")
    parser.add_argument("--video", "-v", required=True, help="Path to input video file or RTSP stream")
    parser.add_argument("--config", "-c", default="config.json", help="Path to configuration file")
    parser.add_argument("--output-log", "-o", default="cctv_activity.log", help="Path to output text log")
    parser.add_argument("--json-log", "-j", default="cctv_activity.jsonl", help="Path to output JSONL log")
    parser.add_argument("--save-video", "-s", help="Optional path to export annotated MP4 video")
    parser.add_argument("--display", "-d", action="store_true", help="Display live preview window")
    parser.add_argument("--device", default="auto", help="Compute device ('cuda', 'cpu', or 'auto')")

    args = parser.parse_args()

    engine = NeuralCCTVEngine(config_path=args.config, device=args.device)
    engine.process_video(
        video_path=args.video,
        output_log_path=args.output_log,
        json_log_path=args.json_log,
        save_video_path=args.save_video,
        display=args.display
    )


if __name__ == "__main__":
    main()
