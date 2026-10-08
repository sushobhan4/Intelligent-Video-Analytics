"""
================================================================================
IBVAP - Intelligent Border Video Analytics Platform
Border Surveillance AI Pipeline for Existing CCTV Infrastructure
================================================================================

Capabilities:
  - Ingests standard IP CCTV / video files without proprietary hardware
  - Human detection and multi-object tracking (ByteTrack / YOLO)
  - Vehicle detection, classification (Car, Truck, Bus, Motorcycle), and tracking
  - Face detection on detected personnel in proximity
  - Automatic Number Plate Recognition (ANPR) for tracked vehicles
  - Virtual fence tripwire intrusion detection (directional crossing)
  - Polygonal restricted zone breach detection
  - Suspicious behavioral analytics:
      * Loitering detection
      * Abnormal rapid movement / sprinting
      * Stationary / abandoned vehicles in sensitive zones
  - Night-time low-light detection & CLAHE enhancement
  - Real-time time-coded event logging (Text .log and JSONL)
  - Optional visual overlay rendering (--save-video / --display)
"""

import os
import sys
import time
import json
import math
import re
import argparse
from datetime import datetime, timedelta
from typing import Dict, List, Tuple, Optional, Any

import cv2
import numpy as np

# Suppress verbose Ultralytics and Torch logs where possible
os.environ["YOLO_VERBOSE"] = "False"
from ultralytics import YOLO

try:
    import easyocr
    EASYOCR_AVAILABLE = True
except ImportError:
    EASYOCR_AVAILABLE = False


# ==============================================================================
# 1. TIME-CODED EVENT LOGGER
# ==============================================================================

class TimeCodedLogger:
    """
    Standardized, security-grade time-coded event logging system.
    Outputs human-readable text logs, JSONL structured events, and formatted console logs.
    """

    SEVERITY_COLORS = {
        "INFO": "\033[94m",      # Blue
        "WARNING": "\033[93m",   # Yellow
        "CRITICAL": "\033[91m",  # Red
        "RESET": "\033[0m"
    }

    def __init__(self, log_file: str, json_file: Optional[str] = None, camera_id: str = "CAM_DEFAULT", log_to_console: bool = True):
        self.log_file = log_file
        self.json_file = json_file
        self.camera_id = camera_id
        self.log_to_console = log_to_console
        self.event_counts = {
            "CRITICAL": 0,
            "WARNING": 0,
            "INFO": 0,
            "TOTAL": 0
        }
        self.category_counts: Dict[str, int] = {}
        self.start_wall_time = datetime.now()

        # Initialize text log file with header
        os.makedirs(os.path.dirname(os.path.abspath(self.log_file)) or ".", exist_ok=True)
        with open(self.log_file, "w", encoding="utf-8") as f:
            f.write("=" * 110 + "\n")
            f.write(" " * 28 + "IBVAP - INTELLIGENT BORDER VIDEO ANALYTICS PLATFORM\n")
            f.write(" " * 36 + "TIME-CODED SURVEILLANCE AUDIT LOG\n")
            f.write("=" * 110 + "\n")
            f.write(f"Session Start: {self.start_wall_time.strftime('%Y-%m-%d %H:%M:%S.%f')[:-3]}\n")
            f.write(f"Camera ID    : {self.camera_id}\n")
            f.write("-" * 110 + "\n")
            f.write(f"{'WALL CLOCK':<24} | {'TIMECODE':<12} | {'FRAME':<7} | {'SEVERITY':<8} | {'EVENT TYPE':<24} | DETAILS\n")
            f.write("-" * 110 + "\n")

        # Initialize JSONL file if requested
        if self.json_file:
            os.makedirs(os.path.dirname(os.path.abspath(self.json_file)) or ".", exist_ok=True)
            with open(self.json_file, "w", encoding="utf-8") as f:
                header = {
                    "ibvap_version": "1.0.0",
                    "session_start": self.start_wall_time.isoformat(),
                    "camera_id": self.camera_id,
                    "event_schema": ["timestamp", "video_timecode", "frame_idx", "severity", "category", "entity", "details"]
                }
                f.write(json.dumps({"_header": header}) + "\n")

    @staticmethod
    def format_timecode(frame_idx: int, fps: float) -> str:
        """Converts frame index and FPS to standard SMPTE-style timecode HH:MM:SS.mmm"""
        if fps <= 0:
            fps = 30.0
        total_seconds = frame_idx / fps
        hours = int(total_seconds // 3600)
        minutes = int((total_seconds % 3600) // 60)
        seconds = int(total_seconds % 60)
        millis = int((total_seconds - int(total_seconds)) * 1000)
        return f"{hours:02d}:{minutes:02d}:{seconds:02d}.{millis:03d}"

    @staticmethod
    def _json_default(obj):
        if isinstance(obj, (np.integer, np.int32, np.int64)):
            return int(obj)
        elif isinstance(obj, (np.floating, np.float32, np.float64)):
            return float(obj)
        elif isinstance(obj, np.ndarray):
            return obj.tolist()
        return str(obj)

    def log_event(self,
                  frame_idx: int,
                  fps: float,
                  severity: str,
                  category: str,
                  entity_type: str,
                  track_id: Optional[int],
                  details: str,
                  bbox: Optional[Tuple[int, int, int, int]] = None,
                  confidence: Optional[float] = None,
                  extra_metadata: Optional[Dict[str, Any]] = None):
        """
        Record an analytic event with precise timecoding and structured metadata.
        """
        timecode = self.format_timecode(frame_idx, fps)
        now_dt = datetime.now()
        wall_str = now_dt.strftime("%Y-%m-%d %H:%M:%S.%f")[:-3]

        self.event_counts[severity] = self.event_counts.get(severity, 0) + 1
        self.event_counts["TOTAL"] += 1
        self.category_counts[category] = self.category_counts.get(category, 0) + 1

        clean_track_id = int(track_id) if track_id is not None else None
        clean_bbox = [int(v) for v in bbox] if bbox is not None else None

        entity_str = f"{entity_type}" + (f" [ID #{clean_track_id}]" if clean_track_id is not None else "")
        full_details = f"{entity_str}: {details}"
        if confidence is not None:
            full_details += f" (Conf: {confidence:.2f})"
        if clean_bbox is not None:
            full_details += f" [BBox: {clean_bbox[0]},{clean_bbox[1]},{clean_bbox[2]},{clean_bbox[3]}]"

        log_line = f"{wall_str:<24} | {timecode:<12} | {frame_idx:<7} | {severity:<8} | {category:<24} | {full_details}\n"

        # Write to human-readable text log
        with open(self.log_file, "a", encoding="utf-8") as f:
            f.write(log_line)

        # Write to JSONL
        if self.json_file:
            event_obj = {
                "wall_time": now_dt.isoformat(),
                "video_timecode": timecode,
                "frame_idx": int(frame_idx),
                "camera_id": self.camera_id,
                "severity": severity,
                "category": category,
                "entity_type": entity_type,
                "track_id": clean_track_id,
                "confidence": round(float(confidence), 3) if confidence is not None else None,
                "bbox": clean_bbox,
                "details": details,
                "extra": extra_metadata or {}
            }
            with open(self.json_file, "a", encoding="utf-8") as f:
                f.write(json.dumps(event_obj, default=self._json_default) + "\n")

        # Console output
        if self.log_to_console:
            color = self.SEVERITY_COLORS.get(severity, "")
            reset = self.SEVERITY_COLORS["RESET"]
            print(f"[{timecode}] {color}[{severity}]{reset} [{category}] {full_details}")

    def close(self, total_frames: int, fps: float, unique_persons: int, unique_vehicles: int):
        """Append summary report to audit log"""
        duration_sec = total_frames / max(fps, 1.0)
        end_wall_time = datetime.now()

        summary = [
            "\n" + "=" * 110,
            " " * 38 + "SURVEILLANCE ANALYTICS SESSION SUMMARY",
            "=" * 110,
            f"Camera ID             : {self.camera_id}",
            f"Session Ended         : {end_wall_time.strftime('%Y-%m-%d %H:%M:%S.%f')[:-3]}",
            f"Elapsed Processing    : {str(end_wall_time - self.start_wall_time)}",
            f"Total Video Frames    : {total_frames} ({self.format_timecode(total_frames, fps)})",
            f"Processed Video Length: {duration_sec:.2f} seconds",
            "-" * 110,
            f"Unique Humans Tracked : {unique_persons}",
            f"Unique Vehicles       : {unique_vehicles}",
            f"Critical Alerts       : {self.event_counts.get('CRITICAL', 0)}",
            f"Warnings Triggered    : {self.event_counts.get('WARNING', 0)}",
            f"Information Events    : {self.event_counts.get('INFO', 0)}",
            f"Total Logged Events   : {self.event_counts['TOTAL']}",
            "-" * 110,
            "Event Breakdown by Category:"
        ]
        for cat, count in sorted(self.category_counts.items(), key=lambda x: -x[1]):
            summary.append(f"  * {cat:<30}: {count} events")
        summary.append("=" * 110 + "\n")

        summary_text = "\n".join(summary)
        with open(self.log_file, "a", encoding="utf-8") as f:
            f.write(summary_text)

        if self.log_to_console:
            print(summary_text)


# ==============================================================================
# 2. GEOMETRIC UTILITIES (VIRTUAL FENCE & INTRUSION DETECTION)
# ==============================================================================

class GeometryUtils:
    """Mathematical and geometric utilities for tripwires and polygonal zones."""

    @staticmethod
    def ccw(A: Tuple[float, float], B: Tuple[float, float], C: Tuple[float, float]) -> bool:
        """Check if three points are listed in counterclockwise order."""
        return (C[1] - A[1]) * (B[0] - A[0]) > (B[1] - A[1]) * (C[0] - A[0])

    @classmethod
    def line_segments_intersect(cls, p1: Tuple[float, float], p2: Tuple[float, float],
                                q1: Tuple[float, float], q2: Tuple[float, float]) -> bool:
        """Determines if segment p1-p2 intersects segment q1-q2."""
        return (cls.ccw(p1, q1, q2) != cls.ccw(p2, q1, q2)) and \
               (cls.ccw(p1, p2, q1) != cls.ccw(p1, p2, q2))

    @staticmethod
    def crossing_direction(prev_pt: Tuple[float, float], curr_pt: Tuple[float, float],
                           line_start: Tuple[float, float], line_end: Tuple[float, float]) -> str:
        """
        Determines the crossing side/vector relative to the directed virtual fence line.
        """
        # Vector line: (dx, dy)
        dx = line_end[0] - line_start[0]
        dy = line_end[1] - line_start[1]
        # Cross product of line vector and movement vector
        mv_x = curr_pt[0] - prev_pt[0]
        mv_y = curr_pt[1] - prev_pt[1]
        cross = dx * mv_y - dy * mv_x
        return "Positive (Inbound)" if cross > 0 else "Negative (Outbound)"

    @staticmethod
    def is_point_in_polygon(pt: Tuple[float, float], polygon: List[Tuple[float, float]]) -> bool:
        """
        Checks if point is inside a polygon using OpenCV pointPolygonTest.
        """
        pts = np.array(polygon, dtype=np.int32)
        dist = cv2.pointPolygonTest(pts, (float(pt[0]), float(pt[1])), False)
        return dist >= 0


# ==============================================================================
# 3. NIGHT-TIME / LOW-LIGHT ADAPTIVE ENHANCEMENT
# ==============================================================================

class NightVisionModule:
    """
    Monitors ambient luminance; dynamically applies CLAHE low-light enhancement.
    """

    def __init__(self, low_light_threshold: float = 65.0, clahe_clip_limit: float = 2.5):
        self.low_light_threshold = low_light_threshold
        self.clahe = cv2.createCLAHE(clipLimit=clahe_clip_limit, tileGridSize=(8, 8))
        self.is_night_mode = False

    def check_illumination(self, frame: np.ndarray) -> Tuple[bool, float]:
        """Calculates mean luminance from LAB color space L-channel."""
        # Convert to LAB to decouple luminance from chrominance
        lab = cv2.cvtColor(frame, cv2.COLOR_BGR2LAB)
        l_channel = lab[:, :, 0]
        mean_lum = float(np.mean(l_channel))
        is_night = mean_lum < self.low_light_threshold
        self.is_night_mode = is_night
        return is_night, mean_lum

    def enhance(self, frame: np.ndarray) -> np.ndarray:
        """Applies CLAHE on luminance channel to restore visibility in shadow regions."""
        lab = cv2.cvtColor(frame, cv2.COLOR_BGR2LAB)
        l, a, b = cv2.split(lab)
        l_enhanced = self.clahe.apply(l)
        enhanced_lab = cv2.merge((l_enhanced, a, b))
        return cv2.cvtColor(enhanced_lab, cv2.COLOR_LAB2BGR)


# ==============================================================================
# 4. FACE DETECTION MODULE
# ==============================================================================

class FaceDetectionModule:
    """
    Lightweight, offline face detector leveraging OpenCV Haar Cascades
    focused on personnel upper bodies / head crops.
    """

    def __init__(self, min_human_height_px: int = 50, scale_factor: float = 1.10, min_neighbors: int = 4):
        frontal_path = cv2.data.haarcascades + 'haarcascade_frontalface_default.xml'
        profile_path = cv2.data.haarcascades + 'haarcascade_profileface.xml'
        self.frontal_cascade = cv2.CascadeClassifier(frontal_path)
        self.profile_cascade = cv2.CascadeClassifier(profile_path)
        self.min_human_height_px = min_human_height_px
        self.scale_factor = scale_factor
        self.min_neighbors = min_neighbors

    def detect_faces(self, frame: np.ndarray, person_bbox: Tuple[int, int, int, int]) -> List[Tuple[int, int, int, int]]:
        """
        Extracts face from upper body region of a detected person.
        Checks both frontal and profile facial orientations.
        Returns list of face bounding boxes in full-frame coordinates [x1, y1, x2, y2].
        """
        x1, y1, x2, y2 = person_bbox
        w = x2 - x1
        h = y2 - y1

        if h < self.min_human_height_px or w <= 10:
            return []

        # Face is typically located in top 45% of human bounding box
        head_y2 = y1 + int(h * 0.45)
        head_roi = frame[max(0, y1):head_y2, max(0, x1):x2]

        if head_roi.size == 0:
            return []

        gray = cv2.cvtColor(head_roi, cv2.COLOR_BGR2GRAY)
        min_size = (max(14, int(w * 0.15)), max(14, int(h * 0.12)))
        faces = list(self.frontal_cascade.detectMultiScale(
            gray,
            scaleFactor=self.scale_factor,
            minNeighbors=self.min_neighbors,
            minSize=min_size
        ))

        if len(faces) == 0:
            faces = list(self.profile_cascade.detectMultiScale(
                gray,
                scaleFactor=self.scale_factor,
                minNeighbors=self.min_neighbors,
                minSize=min_size
            ))

        global_faces = []
        for (fx, fy, fw, fh) in faces:
            gx1 = x1 + fx
            gy1 = y1 + fy
            gx2 = gx1 + fw
            gy2 = gy1 + fh
            global_faces.append((gx1, gy1, gx2, gy2))

        return global_faces


# ==============================================================================
# 5. AUTOMATIC NUMBER PLATE RECOGNITION (ANPR) MODULE
# ==============================================================================

class ANPRModule:
    """
    ANPR pipeline:
      1. Isolates vehicle bumper / lower half region.
      2. Morphological gradient & thresholding to find rectangular plate candidates.
      3. OCR text extraction via EasyOCR.
      4. Alphanumeric filtering and regex verification.
    """

    def __init__(self, min_chars: int = 4, max_chars: int = 12, regex_pattern: str = r"^[A-Z0-9]{4,12}$"):
        self.min_chars = min_chars
        self.max_chars = max_chars
        self.pattern = re.compile(regex_pattern)
        self.ocr_reader = None
        if EASYOCR_AVAILABLE:
            try:
                # Initialize English OCR reader on GPU or CPU
                self.ocr_reader = easyocr.Reader(['en'], gpu=True, verbose=False)
            except Exception as e:
                print(f"[ANPR Warning] EasyOCR GPU failed ({e}), falling back to CPU...")
                try:
                    self.ocr_reader = easyocr.Reader(['en'], gpu=False, verbose=False)
                except Exception as e2:
                    print(f"[ANPR Error] EasyOCR initialization failed: {e2}")

        # Cache of detected plates per vehicle track ID: {track_id: {"plate": text, "conf": float}}
        self.vehicle_plates: Dict[int, Dict[str, Any]] = {}

    def extract_plate_candidates(self, vehicle_roi: np.ndarray) -> List[np.ndarray]:
        """
        Uses morphological edge detection to locate candidate plate crops.
        """
        h, w = vehicle_roi.shape[:2]
        if h < 40 or w < 60:
            return [vehicle_roi]

        # Vehicle license plates are usually located in the lower 60%
        bumper_crop = vehicle_roi[int(h * 0.4):, :]
        candidates = [bumper_crop]

        gray = cv2.cvtColor(bumper_crop, cv2.COLOR_BGR2GRAY)
        # Morphological gradient to emphasize vertical contrast edges
        sobelx = cv2.Sobel(gray, cv2.CV_8U, 1, 0, ksize=3)
        _, thresh = cv2.threshold(sobelx, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
        kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (17, 3))
        morph = cv2.morphologyEx(thresh, cv2.MORPH_CLOSE, kernel)

        contours, _ = cv2.findContours(morph, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        for cnt in contours:
            cx, cy, cw, ch = cv2.boundingRect(cnt)
            aspect_ratio = cw / float(max(ch, 1))
            # License plates standard aspect ratio is between 2.0 and 5.5
            if 2.0 <= aspect_ratio <= 5.5 and cw > 45 and ch > 12:
                plate_crop = bumper_crop[cy:cy + ch, cx:cx + cw]
                if plate_crop.size > 0:
                    candidates.append(plate_crop)

        return candidates

    def recognize_plate(self, frame: np.ndarray, vehicle_bbox: Tuple[int, int, int, int], track_id: int) -> Optional[Tuple[str, float]]:
        """
        Executes OCR on vehicle region and returns cleaned license plate string.
        """
        # If plate already found with high confidence (>0.85), avoid redundant processing
        cached = self.vehicle_plates.get(track_id)
        if cached and cached.get("conf", 0) >= 0.85:
            return None

        if not self.ocr_reader:
            return None

        x1, y1, x2, y2 = vehicle_bbox
        x1, y1 = max(0, x1), max(0, y1)
        x2, y2 = min(frame.shape[1], x2), min(frame.shape[0], y2)
        vehicle_roi = frame[y1:y2, x1:x2]

        if vehicle_roi.size == 0:
            return None

        candidates = self.extract_plate_candidates(vehicle_roi)

        best_plate = None
        best_conf = 0.0

        for crop in candidates[:3]:  # Test top candidates
            # Preprocess crop: resize, grayscale, contrast stretch
            h, w = crop.shape[:2]
            if w < 100:
                scale = 140.0 / max(w, 1)
                crop_proc = cv2.resize(crop, (0, 0), fx=scale, fy=scale, interpolation=cv2.INTER_CUBIC)
            else:
                crop_proc = crop

            try:
                results = self.ocr_reader.readtext(crop_proc, detail=1, allowlist='0123456789ABCDEFGHIJKLMNOPQRSTUVWXYZ')
                for (_, text, score) in results:
                    cleaned = re.sub(r"[^A-Z0-9]", "", text.upper())
                    if self.min_chars <= len(cleaned) <= self.max_chars:
                        if self.pattern.match(cleaned) and score > best_conf:
                            best_plate = cleaned
                            best_conf = float(score)
            except Exception:
                continue

        if best_plate and best_conf > 0.40:
            # Check if this is an improvement over cached
            if not cached or best_conf > cached.get("conf", 0.0):
                self.vehicle_plates[track_id] = {"plate": best_plate, "conf": best_conf}
                return best_plate, best_conf

        return None


# ==============================================================================
# 6. BEHAVIORAL & SUSPICIOUS ACTIVITY ANALYTICS ENGINE
# ==============================================================================

class TrackHistory:
    """Stores movement trajectory, timestamps, and behavioral metrics for a single track."""

    def __init__(self, track_id: int, entity_type: str, initial_pos: Tuple[float, float], timestamp: float):
        self.track_id = track_id
        self.entity_type = entity_type
        self.positions: List[Tuple[float, Tuple[float, float]]] = [(timestamp, initial_pos)]  # (t, (x, y))
        self.first_seen = timestamp
        self.last_seen = timestamp
        self.last_alert_time: Dict[str, float] = {}  # {event_type: last_alert_sec}
        self.crossed_fence = False
        self.breached_zones: set = set()
        self.is_loitering_alerted = False
        self.is_speed_alerted = False
        self.is_stationary_alerted = False

    def update(self, pos: Tuple[float, float], timestamp: float):
        self.positions.append((timestamp, pos))
        self.last_seen = timestamp
        # Retain last 150 points (~5-10 seconds history)
        if len(self.positions) > 150:
            self.positions.pop(0)

    def dwell_time(self) -> float:
        return self.last_seen - self.first_seen

    def get_current_velocity(self) -> float:
        """Calculates instantaneous speed (pixels/second) over the last ~1 second."""
        if len(self.positions) < 2:
            return 0.0
        now_t, now_pt = self.positions[-1]
        # Look back up to 1 second
        for t, pt in reversed(self.positions[:-1]):
            dt = now_t - t
            if dt >= 0.5:
                dx = now_pt[0] - pt[0]
                dy = now_pt[1] - pt[1]
                dist = math.hypot(dx, dy)
                return dist / dt
        return 0.0

    def get_net_displacement(self, duration_sec: float = 5.0) -> float:
        """Returns distance between current position and position N seconds ago."""
        if len(self.positions) < 2:
            return 0.0
        now_t, now_pt = self.positions[-1]
        for t, pt in reversed(self.positions[:-1]):
            if (now_t - t) >= duration_sec:
                return math.hypot(now_pt[0] - pt[0], now_pt[1] - pt[1])
        # If not enough history, displacement from beginning
        return math.hypot(now_pt[0] - self.positions[0][1][0], now_pt[1] - self.positions[0][1][1])

    def can_alert(self, alert_type: str, current_time: float, cooldown: float) -> bool:
        last = self.last_alert_time.get(alert_type, -9999.0)
        if current_time - last >= cooldown:
            self.last_alert_time[alert_type] = current_time
            return True
        return False


class SuspiciousActivityEngine:
    """
    Evaluates tracking data to identify security breaches and behavioral anomalies:
      - Virtual Fence Tripwire Crossings
      - Polygonal Restricted Zone Intrusions
      - Loitering in sensitive perimeters
      - Abnormal high-velocity sprints / rushes
      - Stationary / abandoned vehicles
    """

    def __init__(self, config: Dict[str, Any], logger: TimeCodedLogger, frame_w: int, frame_h: int):
        self.config = config
        self.logger = logger
        self.frame_w = frame_w
        self.frame_h = frame_h
        self.tracks: Dict[int, TrackHistory] = {}

        # Parse virtual fence line
        vf_cfg = config.get("virtual_fence", {})
        self.fence_enabled = vf_cfg.get("enabled", True)
        norm_line = vf_cfg.get("line", [[0.1, 0.65], [0.9, 0.65]])
        self.fence_p1 = (int(norm_line[0][0] * frame_w), int(norm_line[0][1] * frame_h))
        self.fence_p2 = (int(norm_line[1][0] * frame_w), int(norm_line[1][1] * frame_h))
        self.dir_pos_label = vf_cfg.get("direction_label_positive", "Inbound (Restricted Crossing)")
        self.dir_neg_label = vf_cfg.get("direction_label_negative", "Outbound")

        # Parse restricted zones
        self.zones = []
        for z in config.get("restricted_zones", []):
            poly = [(int(p[0] * frame_w), int(p[1] * frame_h)) for p in z.get("polygon", [])]
            self.zones.append({
                "id": z.get("zone_id", "ZONE_UNKNOWN"),
                "name": z.get("zone_name", "Restricted Area"),
                "poly": poly,
                "severity": z.get("severity", "CRITICAL")
            })

        # Thresholds
        rules = config.get("behavioral_rules", {})
        self.loiter_thresh = rules.get("loitering_threshold_seconds", 5.0)
        self.speed_thresh = rules.get("rapid_movement_speed_threshold", 250.0)
        self.stationary_thresh = rules.get("stationary_vehicle_threshold_seconds", 8.0)
        self.cooldown = rules.get("alert_cooldown_seconds", 3.0)

    def update_track(self, track_id: int, entity_type: str, bbox: Tuple[int, int, int, int],
                     frame_idx: int, fps: float, confidence: float, is_night: bool):
        current_time_sec = frame_idx / max(fps, 1.0)
        x1, y1, x2, y2 = bbox
        centroid = ((x1 + x2) / 2.0, y2)  # Use bottom center (feet/ground contact) for accurate fence crossing

        if track_id not in self.tracks:
            self.tracks[track_id] = TrackHistory(track_id, entity_type, centroid, current_time_sec)
            # Log initial detection
            self.logger.log_event(
                frame_idx=frame_idx,
                fps=fps,
                severity="INFO",
                category="DETECTION_INITIAL",
                entity_type=entity_type,
                track_id=track_id,
                details=f"New {entity_type} entered surveillance corridor at position ({int(centroid[0])}, {int(centroid[1])})",
                bbox=bbox,
                confidence=confidence,
                extra_metadata={"is_night": is_night}
            )
        else:
            self.tracks[track_id].update(centroid, current_time_sec)

        track = self.tracks[track_id]

        # ----------------------------------------------------------------------
        # A. VIRTUAL FENCE TRIPWIRE CROSSING
        # ----------------------------------------------------------------------
        if self.fence_enabled and len(track.positions) >= 2:
            prev_pt = track.positions[-2][1]
            curr_pt = centroid
            if GeometryUtils.line_segments_intersect(prev_pt, curr_pt, self.fence_p1, self.fence_p2):
                should_alert = False
                if not track.crossed_fence:
                    track.crossed_fence = True
                    track.last_alert_time["FENCE_BREACH"] = current_time_sec
                    should_alert = True
                elif track.can_alert("FENCE_BREACH", current_time_sec, self.cooldown):
                    should_alert = True

                if should_alert:
                    direction = GeometryUtils.crossing_direction(prev_pt, curr_pt, self.fence_p1, self.fence_p2)
                    dir_str = self.dir_pos_label if "Positive" in direction else self.dir_neg_label
                    self.logger.log_event(
                        frame_idx=frame_idx,
                        fps=fps,
                        severity="CRITICAL",
                        category="VIRTUAL_FENCE_BREACH",
                        entity_type=entity_type,
                        track_id=track_id,
                        details=f"Intruder breached Border Virtual Fence line! Vector: {dir_str}",
                        bbox=bbox,
                        confidence=confidence,
                        extra_metadata={"direction": dir_str, "crossing_point": [int(curr_pt[0]), int(curr_pt[1])]}
                    )

        # ----------------------------------------------------------------------
        # B. RESTRICTED POLYGONAL ZONE INTRUSION
        # ----------------------------------------------------------------------
        for zone in self.zones:
            if GeometryUtils.is_point_in_polygon(centroid, zone["poly"]):
                should_alert = False
                if zone["id"] not in track.breached_zones:
                    track.breached_zones.add(zone["id"])
                    track.last_alert_time[f"ZONE_{zone['id']}"] = current_time_sec
                    should_alert = True
                elif track.can_alert(f"ZONE_{zone['id']}", current_time_sec, self.cooldown):
                    should_alert = True

                if should_alert:
                    self.logger.log_event(
                        frame_idx=frame_idx,
                        fps=fps,
                        severity=zone["severity"],
                        category="RESTRICTED_ZONE_INTRUSION",
                        entity_type=entity_type,
                        track_id=track_id,
                        details=f"Intrusion detected in {zone['name']} [{zone['id']}]",
                        bbox=bbox,
                        confidence=confidence,
                        extra_metadata={"zone_id": zone["id"], "zone_name": zone["name"]}
                    )

        # ----------------------------------------------------------------------
        # C. BEHAVIORAL: LOITERING DETECTION (HUMAN)
        # ----------------------------------------------------------------------
        if entity_type == "Person":
            dwell = track.dwell_time()
            if dwell >= self.loiter_thresh:
                net_disp = track.get_net_displacement(duration_sec=self.loiter_thresh)
                # If movement radius is less than 90 pixels over the dwelling duration
                if net_disp < 90.0:
                    if track.can_alert("LOITERING", current_time_sec, self.cooldown * 2):
                        self.logger.log_event(
                            frame_idx=frame_idx,
                            fps=fps,
                            severity="WARNING",
                            category="SUSPICIOUS_LOITERING",
                            entity_type=entity_type,
                            track_id=track_id,
                            details=f"Suspicious loitering behavior detected! Dwell time: {dwell:.1f}s, Net displacement: {net_disp:.1f}px",
                            bbox=bbox,
                            confidence=confidence,
                            extra_metadata={"dwell_time_sec": round(dwell, 1), "net_displacement_px": round(net_disp, 1)}
                        )

        # ----------------------------------------------------------------------
        # D. BEHAVIORAL: RAPID MOVEMENT / SPRINTING (HUMAN)
        # ----------------------------------------------------------------------
        if entity_type == "Person":
            speed = track.get_current_velocity()
            if speed > self.speed_thresh:
                if track.can_alert("RAPID_SPEED", current_time_sec, self.cooldown):
                    self.logger.log_event(
                        frame_idx=frame_idx,
                        fps=fps,
                        severity="WARNING",
                        category="SUSPICIOUS_RAPID_MOVEMENT",
                        entity_type=entity_type,
                        track_id=track_id,
                        details=f"Abnormal rapid movement / border rush sprint detected! Speed: {speed:.1f} px/sec",
                        bbox=bbox,
                        confidence=confidence,
                        extra_metadata={"speed_px_sec": round(speed, 1)}
                    )

        # ----------------------------------------------------------------------
        # E. BEHAVIORAL: STATIONARY / ABANDONED VEHICLE
        # ----------------------------------------------------------------------
        if entity_type in ["Car", "Truck", "Bus", "Motorcycle"]:
            dwell = track.dwell_time()
            if dwell >= self.stationary_thresh:
                net_disp = track.get_net_displacement(duration_sec=self.stationary_thresh)
                if net_disp < 35.0:  # Vehicle hasn't moved
                    if track.can_alert("STATIONARY_VEHICLE", current_time_sec, self.cooldown * 2):
                        self.logger.log_event(
                            frame_idx=frame_idx,
                            fps=fps,
                            severity="WARNING",
                            category="SUSPICIOUS_STATIONARY_VEHICLE",
                            entity_type=entity_type,
                            track_id=track_id,
                            details=f"Stationary / abandoned vehicle in border corridor! Static for {dwell:.1f}s",
                            bbox=bbox,
                            confidence=confidence,
                            extra_metadata={"dwell_sec": round(dwell, 1)}
                        )

        # ----------------------------------------------------------------------
        # F. NIGHT-TIME MOVEMENT ALERT
        # ----------------------------------------------------------------------
        if is_night:
            if track.can_alert("NIGHT_MOVEMENT", current_time_sec, self.cooldown * 3):
                self.logger.log_event(
                    frame_idx=frame_idx,
                    fps=fps,
                    severity="WARNING",
                    category="NIGHT_TIME_MOVEMENT",
                    entity_type=entity_type,
                    track_id=track_id,
                    details=f"Movement detected under low-light/night conditions at ({int(centroid[0])}, {int(centroid[1])})",
                    bbox=bbox,
                    confidence=confidence,
                    extra_metadata={"is_night": True}
                )


# ==============================================================================
# 7. MAIN VIDEO ANALYTICS PLATFORM PIPELINE (IBVAP)
# ==============================================================================

class IBVAPEngine:
    """
    Intelligent Border Video Analytics Platform core inference engine.
    Ingests video, tracks humans & vehicles, executes ANPR, face detection,
    monitors virtual fence & behavioral anomalies, and writes time-coded logs.
    """

    # Class mapping for standard COCO / YOLO model
    COCO_CLASS_MAP = {
        0: "Person",
        1: "Bicycle",
        2: "Car",
        3: "Motorcycle",
        5: "Bus",
        7: "Truck"
    }

    def __init__(self,
                 video_source: str,
                 output_log: str = "ibvap_events.log",
                 json_log: Optional[str] = "ibvap_events.jsonl",
                 config_path: Optional[str] = None,
                 save_video: Optional[str] = None,
                 display: bool = False,
                 device: Optional[str] = None,
                 conf_threshold: Optional[float] = None):
        self.video_source = video_source
        self.output_log = output_log
        self.json_log = json_log
        self.save_video = save_video
        self.display = display

        # Load configuration
        self.config = self._load_config(config_path)

        camera_cfg = self.config.get("camera", {})
        self.camera_id = camera_cfg.get("camera_id", "BOP_CCTV_STREAM_01")

        # Initialize Time-Coded Logger
        log_cfg = self.config.get("logging", {})
        self.logger = TimeCodedLogger(
            log_file=self.output_log,
            json_file=self.json_log if log_cfg.get("json_log", True) else None,
            camera_id=self.camera_id,
            log_to_console=log_cfg.get("log_to_console", True)
        )

        # Video Capture
        self.cap = cv2.VideoCapture(self.video_source)
        if not self.cap.isOpened():
            raise RuntimeError(f"[IBVAP Critical Error] Failed to open video source: {self.video_source}")

        self.width = int(self.cap.get(cv2.CAP_PROP_FRAME_WIDTH))
        self.height = int(self.cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
        self.fps = self.cap.get(cv2.CAP_PROP_FPS)
        if self.fps <= 0 or math.isnan(self.fps):
            self.fps = 30.0
        self.total_frames = int(self.cap.get(cv2.CAP_PROP_FRAME_COUNT))

        self.logger.log_event(
            frame_idx=0,
            fps=self.fps,
            severity="INFO",
            category="STREAM_INITIALIZED",
            entity_type="SYSTEM",
            track_id=None,
            details=f"Ingesting stream: {self.video_source} | Resolution: {self.width}x{self.height} | FPS: {self.fps:.1f} | Total Frames: {self.total_frames}"
        )

        # Device selection
        if device is None:
            analytics_cfg = self.config.get("analytics", {})
            self.device = analytics_cfg.get("device", "cuda" if cv2.cuda.getCudaEnabledDeviceCount() > 0 else "cpu")
        else:
            self.device = device

        # Confidence
        if conf_threshold is None:
            self.conf_threshold = self.config.get("analytics", {}).get("confidence_threshold", 0.35)
        else:
            self.conf_threshold = conf_threshold

        # Load YOLO Model
        model_name = self.config.get("analytics", {}).get("model_path", "yolov8n.pt")
        print(f"[IBVAP] Loading Object Detection Model ({model_name}) on device: {self.device}...")
        self.model = YOLO(model_name)

        # Initialize Sub-modules
        nv_cfg = self.config.get("night_vision", {})
        self.night_module = NightVisionModule(
            low_light_threshold=nv_cfg.get("low_light_threshold", 65.0),
            clahe_clip_limit=nv_cfg.get("clahe_clip_limit", 2.5)
        )

        face_cfg = self.config.get("face_detection", {})
        self.face_detector = FaceDetectionModule(
            min_human_height_px=face_cfg.get("min_human_height_px", 80),
            scale_factor=face_cfg.get("scale_factor", 1.15),
            min_neighbors=face_cfg.get("min_neighbors", 4)
        )

        anpr_cfg = self.config.get("anpr", {})
        self.anpr_module = ANPRModule(
            min_chars=anpr_cfg.get("min_plate_chars", 4),
            max_chars=anpr_cfg.get("max_plate_chars", 12),
            regex_pattern=anpr_cfg.get("regex_pattern", r"^[A-Z0-9]{4,12}$")
        )

        self.activity_engine = SuspiciousActivityEngine(
            config=self.config,
            logger=self.logger,
            frame_w=self.width,
            frame_h=self.height
        )

        # Video Writer (if requested)
        self.writer = None
        if self.save_video:
            os.makedirs(os.path.dirname(os.path.abspath(self.save_video)) or ".", exist_ok=True)
            fourcc = cv2.VideoWriter_fourcc(*"mp4v")
            self.writer = cv2.VideoWriter(self.save_video, fourcc, self.fps, (self.width, self.height))

        # Statistics
        self.unique_persons = set()
        self.unique_vehicles = set()
        self.detected_faces_count = 0
        self.anpr_recognized_count = 0

    def _load_config(self, config_path: Optional[str]) -> Dict[str, Any]:
        """Loads configuration from JSON file or provides default border surveillance params."""
        default_config = {
            "camera": {"camera_id": "BOP_SURVEILLANCE_CAM_01"},
            "analytics": {"confidence_threshold": 0.35, "tracker": "bytetrack.yaml", "model_path": "yolov8n.pt", "device": "cuda"},
            "virtual_fence": {"enabled": True, "line": [[0.1, 0.65], [0.9, 0.65]], "direction_label_positive": "Inbound (Restricted)", "direction_label_negative": "Outbound"},
            "restricted_zones": [{"zone_id": "ZONE_ZERO_LINE", "zone_name": "Zero-Line Restricted Strip", "polygon": [[0.05, 0.55], [0.95, 0.55], [0.95, 0.85], [0.05, 0.85]], "severity": "CRITICAL"}],
            "behavioral_rules": {"loitering_threshold_seconds": 5.0, "rapid_movement_speed_threshold": 250.0, "stationary_vehicle_threshold_seconds": 8.0, "alert_cooldown_seconds": 3.0},
            "night_vision": {"enabled": True, "low_light_threshold": 65.0, "clahe_clip_limit": 2.5},
            "anpr": {"enabled": True, "min_plate_chars": 4, "max_plate_chars": 12, "regex_pattern": "^[A-Z0-9]{4,12}$"},
            "face_detection": {"enabled": True, "min_human_height_px": 80},
            "logging": {"log_to_console": True, "json_log": True}
        }

        if config_path and os.path.exists(config_path):
            try:
                with open(config_path, "r", encoding="utf-8") as f:
                    user_cfg = json.load(f)
                    # Merge shallow configs
                    for k, v in user_cfg.items():
                        if isinstance(v, dict) and k in default_config:
                            default_config[k].update(v)
                        else:
                            default_config[k] = v
                print(f"[IBVAP] Successfully loaded configuration from: {config_path}")
            except Exception as e:
                print(f"[IBVAP Warning] Could not parse config file ({e}). Using default surveillance rules.")

        return default_config

    def run(self):
        """
        Main processing loop: frame ingestion, inference, analytics, and time-coded event dispatch.
        """
        print(f"\n[IBVAP] Starting Border Video Analytics Pipeline on: {self.video_source}")
        print(f"[IBVAP] Audit Log Output -> {self.output_log}")
        if self.json_log:
            print(f"[IBVAP] JSONL Log Output  -> {self.json_log}")
        if self.save_video:
            print(f"[IBVAP] Video Annotated   -> {self.save_video}")

        frame_idx = 0
        target_classes = list(self.COCO_CLASS_MAP.keys())
        prev_night_state = False

        try:
            while self.cap.isOpened():
                ret, frame = self.cap.read()
                if not ret or frame is None:
                    break

                frame_idx += 1

                # 1. Check Illumination / Night Mode
                is_night, mean_lum = self.night_module.check_illumination(frame)
                if is_night != prev_night_state:
                    prev_night_state = is_night
                    severity = "WARNING" if is_night else "INFO"
                    status_text = "ENGAGED (Low-Light Condition)" if is_night else "DISENGAGED (Normal Daylight)"
                    self.logger.log_event(
                        frame_idx=frame_idx,
                        fps=self.fps,
                        severity=severity,
                        category="ENVIRONMENT_CHANGE",
                        entity_type="SYSTEM",
                        track_id=None,
                        details=f"Night-Vision mode {status_text} | Mean Luminance: {mean_lum:.1f}/255"
                    )

                # Preprocess frame if in night mode
                processed_frame = frame
                if is_night and self.config.get("night_vision", {}).get("enabled", True):
                    processed_frame = self.night_module.enhance(frame)

                # 2. Run YOLO Tracking (ByteTrack)
                results = self.model.track(
                    source=processed_frame,
                    persist=True,
                    classes=target_classes,
                    conf=self.conf_threshold,
                    device=self.device,
                    tracker="bytetrack.yaml",
                    verbose=False
                )

                annotated_frame = frame.copy() if (self.save_video or self.display) else None

                # Process detections
                if results and len(results) > 0 and results[0].boxes is not None and results[0].boxes.id is not None:
                    boxes = results[0].boxes.xyxy.cpu().numpy()
                    track_ids = results[0].boxes.id.int().cpu().numpy()
                    class_ids = results[0].boxes.cls.int().cpu().numpy()
                    confs = results[0].boxes.conf.cpu().numpy()

                    for box, track_id, cls_id, conf in zip(boxes, track_ids, class_ids, confs):
                        x1, y1, x2, y2 = map(int, box)
                        entity_type = self.COCO_CLASS_MAP.get(int(cls_id), "Unknown")

                        if entity_type == "Person":
                            self.unique_persons.add(int(track_id))
                        elif entity_type in ["Car", "Truck", "Bus", "Motorcycle"]:
                            self.unique_vehicles.add(int(track_id))

                        # A. Update Behavioral & Intrusion Engine
                        self.activity_engine.update_track(
                            track_id=int(track_id),
                            entity_type=entity_type,
                            bbox=(x1, y1, x2, y2),
                            frame_idx=frame_idx,
                            fps=self.fps,
                            confidence=float(conf),
                            is_night=is_night
                        )

                        # B. Face Detection (if human)
                        if entity_type == "Person" and self.config.get("face_detection", {}).get("enabled", True):
                            faces = self.face_detector.detect_faces(frame, (x1, y1, x2, y2))
                            for fx1, fy1, fx2, fy2 in faces:
                                self.detected_faces_count += 1
                                # Log face detection (throttled inside logger/track history)
                                track = self.activity_engine.tracks.get(int(track_id))
                                curr_sec = frame_idx / self.fps
                                if track and track.can_alert("FACE_DETECTED", curr_sec, 4.0):
                                    self.logger.log_event(
                                        frame_idx=frame_idx,
                                        fps=self.fps,
                                        severity="INFO",
                                        category="FACE_DETECTED",
                                        entity_type="Human Face",
                                        track_id=int(track_id),
                                        details=f"Facial features captured on Person #{track_id} (Size: {fx2-fx1}x{fy2-fy1}px)",
                                        bbox=(fx1, fy1, fx2, fy2),
                                        confidence=float(conf)
                                    )
                                if annotated_frame is not None:
                                    cv2.rectangle(annotated_frame, (fx1, fy1), (fx2, fy2), (255, 105, 180), 2)
                                    cv2.putText(annotated_frame, "Face", (fx1, fy1 - 5),
                                                cv2.FONT_HERSHEY_SIMPLEX, 0.45, (255, 105, 180), 1)

                        # C. ANPR (if vehicle)
                        if entity_type in ["Car", "Truck", "Bus", "Motorcycle"] and self.config.get("anpr", {}).get("enabled", True):
                            plate_res = self.anpr_module.recognize_plate(frame, (x1, y1, x2, y2), int(track_id))
                            if plate_res:
                                plate_text, plate_conf = plate_res
                                self.anpr_recognized_count += 1
                                self.logger.log_event(
                                    frame_idx=frame_idx,
                                    fps=self.fps,
                                    severity="INFO",
                                    category="ANPR_PLATE_RECOGNIZED",
                                    entity_type=entity_type,
                                    track_id=int(track_id),
                                    details=f"License Plate recognized: [{plate_text}] for {entity_type} #{track_id}",
                                    bbox=(x1, y1, x2, y2),
                                    confidence=plate_conf,
                                    extra_metadata={"license_plate": plate_text, "plate_conf": round(plate_conf, 3)}
                                )

                        # Visual annotation for bounding box
                        if annotated_frame is not None:
                            color = (0, 255, 0) if entity_type == "Person" else (255, 191, 0)
                            cv2.rectangle(annotated_frame, (x1, y1), (x2, y2), color, 2)
                            label = f"{entity_type} #{track_id} ({conf:.2f})"
                            # If vehicle has detected plate, display it
                            plate_info = self.anpr_module.vehicle_plates.get(int(track_id))
                            if plate_info:
                                label += f" [{plate_info['plate']}]"
                            cv2.putText(annotated_frame, label, (x1, max(20, y1 - 8)),
                                        cv2.FONT_HERSHEY_SIMPLEX, 0.5, color, 2)

                # Visual overlay for virtual fence and zones
                if annotated_frame is not None:
                    self._draw_overlay(annotated_frame, frame_idx, is_night)
                    if self.writer is not None:
                        self.writer.write(annotated_frame)
                    if self.display:
                        cv2.imshow("IBVAP - Live Border Surveillance Analytics", annotated_frame)
                        if cv2.waitKey(1) & 0xFF == ord('q'):
                            print("[IBVAP] Processing aborted by user ('q' pressed).")
                            break

        finally:
            self.cap.release()
            if self.writer is not None:
                self.writer.release()
            if self.display:
                cv2.destroyAllWindows()

            # Finalize session and write summary
            self.logger.close(
                total_frames=frame_idx,
                fps=self.fps,
                unique_persons=len(self.unique_persons),
                unique_vehicles=len(self.unique_vehicles)
            )

    def _draw_overlay(self, frame: np.ndarray, frame_idx: int, is_night: bool):
        """Draws virtual fence, restricted polygons, timecode HUD, and status banners."""
        # Draw restricted polygonal zones
        for zone in self.activity_engine.zones:
            pts = np.array(zone["poly"], dtype=np.int32)
            cv2.polylines(frame, [pts], isClosed=True, color=(0, 0, 255), thickness=2)
            # Semi-transparent fill
            overlay = frame.copy()
            cv2.fillPoly(overlay, [pts], (0, 0, 180))
            cv2.addWeighted(overlay, 0.20, frame, 0.80, 0, frame)
            # Label
            if len(zone["poly"]) > 0:
                cv2.putText(frame, f"RESTRICTED: {zone['name']}",
                            (zone["poly"][0][0] + 5, zone["poly"][0][1] + 20),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.45, (0, 0, 255), 1)

        # Draw virtual fence line
        if self.activity_engine.fence_enabled:
            p1 = self.activity_engine.fence_p1
            p2 = self.activity_engine.fence_p2
            cv2.line(frame, p1, p2, (0, 255, 255), 3)
            cv2.putText(frame, "--- VIRTUAL BORDER TRIPWIRE ---", (p1[0], max(30, p1[1] - 10)),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.55, (0, 255, 255), 2)

        # Top HUD Banner
        h, w = frame.shape[:2]
        hud_h = 35
        cv2.rectangle(frame, (0, 0), (w, hud_h), (20, 20, 20), -1)

        timecode = TimeCodedLogger.format_timecode(frame_idx, self.fps)
        hud_text = f"CAM: {self.camera_id} | TC: {timecode} | FRAME: {frame_idx}"
        cv2.putText(frame, hud_text, (10, 22), cv2.FONT_HERSHEY_SIMPLEX, 0.55, (255, 255, 255), 1)

        # Night mode indicator
        if is_night:
            cv2.putText(frame, "[NIGHT VISION ACTIVE]", (w - 230, 22),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.55, (0, 255, 128), 2)


# ==============================================================================
# 8. COMMAND LINE INTERFACE (CLI) ENTRY POINT
# ==============================================================================

def main():
    parser = argparse.ArgumentParser(
        description="IBVAP - Intelligent Border Video Analytics Platform (Surveillance AI for CCTV)",
        formatter_class=argparse.RawTextHelpFormatter
    )
    parser.add_argument("--video", "-v", type=str, required=True,
                        help="Path to input video file (e.g. border_cctv.mp4) or RTSP stream URL.")
    parser.add_argument("--output-log", "-o", type=str, default="ibvap_events.log",
                        help="Path to time-coded text audit log file (default: ibvap_events.log)")
    parser.add_argument("--json-log", "-j", type=str, default="ibvap_events.jsonl",
                        help="Path to time-coded JSONL structured log file (default: ibvap_events.jsonl)")
    parser.add_argument("--config", "-c", type=str, default="config.json",
                        help="Path to configuration JSON file (default: config.json)")
    parser.add_argument("--save-video", "-s", type=str, default=None,
                        help="Optional: Path to output annotated MP4 video with security overlays.")
    parser.add_argument("--display", "-d", action="store_true",
                        help="Optional: Display live annotated video window during processing.")
    parser.add_argument("--device", type=str, default=None,
                        help="Device to run inference on: 'cuda' or 'cpu' (default: auto-detect)")
    parser.add_argument("--confidence", type=float, default=None,
                        help="Detection confidence threshold between 0.1 and 1.0 (default: 0.35)")

    args = parser.parse_args()

    if not os.path.exists(args.video) and not args.video.startswith("rtsp://") and not args.video.startswith("http"):
        print(f"[Error] Specified video file does not exist: {args.video}", file=sys.stderr)
        sys.exit(1)

    print("=" * 80)
    print("   IBVAP - INTELLIGENT BORDER VIDEO ANALYTICS PLATFORM")
    print("   AI Surveillance for Existing CCTV Infrastructure")
    print("=" * 80)

    try:
        pipeline = IBVAPEngine(
            video_source=args.video,
            output_log=args.output_log,
            json_log=args.json_log,
            config_path=args.config,
            save_video=args.save_video,
            display=args.display,
            device=args.device,
            conf_threshold=args.confidence
        )
        pipeline.run()
        print(f"\n[IBVAP] Processing complete! Time-coded log written to: {args.output_log}")
        if args.json_log:
            print(f"[IBVAP] Structured JSONL log written to: {args.json_log}")

    except KeyboardInterrupt:
        print("\n[IBVAP] Processing interrupted by user.")
    except Exception as e:
        print(f"\n[IBVAP Error] Execution failed: {e}", file=sys.stderr)
        import traceback
        traceback.print_exc()
        sys.exit(1)


if __name__ == "__main__":
    main()
