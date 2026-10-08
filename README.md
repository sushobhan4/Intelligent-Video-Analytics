# IBVAP: Intelligent Border Video Analytics Platform
### AI-Powered Surveillance Platform for Existing CCTV Infrastructure

IBVAP transforms conventional, standard IP-based CCTV cameras at Border Out Posts (BOPs), check posts, and strategic perimeters into an intelligent automated surveillance network without requiring dedicated proprietary smart cameras or specialized hardware.

---

## 🚀 Capabilities Implemented

| Capability | Module & Implementation | Output in Log |
|---|---|---|
| **Human Detection & Tracking** | YOLOv8 / YOLO11 + ByteTrack persistent multi-object tracking | `DETECTION_INITIAL`, Track IDs, Centroids, BBoxes |
| **Vehicle Detection & Classification** | Deep learning classifier for Cars, Trucks, Buses, Motorcycles | `DETECTION_INITIAL` with Vehicle Classification |
| **Face Detection** | Multi-angle Haar Cascade (Frontal + Profile) on personnel crops | `FACE_DETECTED` with facial bounding box & track association |
| **Automatic Number Plate Recognition (ANPR)** | Morphological bumper extraction + OCR + Alphanumeric regex validation | `ANPR_PLATE_RECOGNIZED` with extracted plate string & score |
| **Virtual Fence Intrusion Detection** | Vector line segment intersection with directional crossing calculation | `VIRTUAL_FENCE_BREACH` (Inbound / Outbound) |
| **Restricted Zone Perimeter Defense** | Polygonal geo-fencing via `cv2.pointPolygonTest` | `RESTRICTED_ZONE_INTRUSION` |
| **Suspicious Activity Analytics** | Behavioral engine monitoring dwell time vs displacement & velocity | `SUSPICIOUS_LOITERING`, `SUSPICIOUS_RAPID_MOVEMENT`, `SUSPICIOUS_STATIONARY_VEHICLE` |
| **Night-Time Movement Detection** | Ambient luminance monitoring + CLAHE low-light enhancement | `ENVIRONMENT_CHANGE` (Night Mode Engaged), `NIGHT_TIME_MOVEMENT` |
| **Time-Coded Event Logging** | Military/Security audit log with SMPTE-style timecode and JSONL | Standardized `.log` file and machine-readable `.jsonl` |

---

## 📁 Project Structure

```
Intelligent-Video-Analytics/
├── ibvap.py                       # Main core analytics platform pipeline (CLI entrypoint)
├── config.json                    # Border perimeter configuration & rule thresholds
├── requirements.txt               # Dependencies
├── .gitignore                     # Ignored artifacts, videos, logs, and weights
├── create_test_video.py           # Synthetic border surveillance video generator (optional)
└── README.md                      # Platform documentation
```

---

## 🛠️ Usage Guide

### Basic Execution (Input Video -> Time-Coded Log)

To analyze any CCTV video file and spit out a time-coded log file:

```bash
python ibvap.py --video path/to/cctv_stream.mp4 --output-log surveillance_events.log
```

### Full Options & Command Line Arguments

```bash
python ibvap.py \
    --video border_cctv_simulation.mp4 \
    --output-log border_events.log \
    --json-log border_events.jsonl \
    --config config.json \
    --save-video border_annotated.mp4 \
    --device cuda \
    --confidence 0.35
```

### CLI Arguments Reference

| Flag | Default | Description |
|---|---|---|
| `--video`, `-v` | *(Required)* | Path to input video file (`.mp4`, `.avi`, `.mkv`) or RTSP stream URL (`rtsp://...`) |
| `--output-log`, `-o` | `ibvap_events.log` | Destination path for the human-readable time-coded audit log |
| `--json-log`, `-j` | `ibvap_events.jsonl` | Optional path for machine-readable JSONL structured event log |
| `--config`, `-c` | `config.json` | JSON configuration file for border zones, fences, and alert thresholds |
| `--save-video`, `-s` | `None` | Optional path to export annotated MP4 video with visual bounding boxes & overlays |
| `--display`, `-d` | `False` | Display live OpenCV preview window in real time (`'q'` to quit) |
| `--device` | Auto (`cuda`/`cpu`) | Processing hardware device (`cuda` for GPU or `cpu`) |
| `--confidence` | `0.35` | Object detection confidence threshold (0.1 to 1.0) |

---

## ⚙️ Configuration (`config.json`)

You can customize border lines and intrusion perimeters for any camera angle by modifying `config.json`:

```json
{
  "camera": {
    "camera_id": "BOP_NORTH_SECTOR_CAM_01",
    "location": "Border Outpost Alpha - Sector 4"
  },
  "virtual_fence": {
    "enabled": true,
    "line": [
      [0.10, 0.65],
      [0.90, 0.65]
    ],
    "direction_label_positive": "Inbound (Restricted Crossing)",
    "direction_label_negative": "Outbound"
  },
  "restricted_zones": [
    {
      "zone_id": "ZONE_BORDER_STRIP",
      "zone_name": "Zero-Line Restricted Strip",
      "polygon": [
        [0.05, 0.55],
        [0.95, 0.55],
        [0.95, 0.85],
        [0.05, 0.85]
      ],
      "severity": "CRITICAL"
    }
  ],
  "behavioral_rules": {
    "loitering_threshold_seconds": 5.0,
    "rapid_movement_speed_threshold": 250.0,
    "stationary_vehicle_threshold_seconds": 8.0,
    "alert_cooldown_seconds": 3.0
  },
  "night_vision": {
    "enabled": true,
    "low_light_threshold": 65.0,
    "clahe_clip_limit": 2.5
  },
  "anpr": {
    "enabled": true,
    "min_plate_chars": 4,
    "max_plate_chars": 12,
    "regex_pattern": "^[A-Z0-9]{4,12}$"
  }
}
```

---

## 📊 Sample Time-Coded Log Output

### 1. Human-Readable Audit Log (`border_events.log`)

```
==============================================================================================================
                            IBVAP - INTELLIGENT BORDER VIDEO ANALYTICS PLATFORM
                                    TIME-CODED SURVEILLANCE AUDIT LOG
==============================================================================================================
Session Start: 2026-10-08 22:47:24.858
Camera ID    : BOP_NORTH_SECTOR_CAM_01
--------------------------------------------------------------------------------------------------------------
WALL CLOCK               | TIMECODE     | FRAME   | SEVERITY | EVENT TYPE               | DETAILS
--------------------------------------------------------------------------------------------------------------
2026-10-08 22:47:24.863  | 00:00:00.000 | 0       | INFO     | STREAM_INITIALIZED       | SYSTEM: Ingesting stream: border_cctv_simulation.mp4 | Resolution: 1280x720 | FPS: 30.0 | Total Frames: 360
2026-10-08 22:47:35.390  | 00:00:00.033 | 1       | INFO     | DETECTION_INITIAL        | Person [ID #1]: New Person entered surveillance corridor at position (925, 481) (Conf: 0.87) [BBox: 878,229,972,481]
2026-10-08 22:47:35.391  | 00:00:00.033 | 1       | CRITICAL | RESTRICTED_ZONE_INTRUSION | Person [ID #1]: Intrusion detected in Zero-Line Restricted Strip [ZONE_BORDER_STRIP] (Conf: 0.87) [BBox: 878,229,972,481]
2026-10-08 22:47:41.073  | 00:00:03.833 | 115     | INFO     | ANPR_PLATE_RECOGNIZED    | Bus [ID #4]: License Plate recognized: [DLO1AB9876] for Bus #4 (Conf: 0.95) [BBox: 292,307,595,508]
2026-10-08 22:47:42.612  | 00:00:05.033 | 151     | WARNING  | SUSPICIOUS_LOITERING     | Person [ID #1]: Suspicious loitering behavior detected! Dwell time: 5.0s, Net displacement: 2.8px (Conf: 0.88) [BBox: 876,230,971,480]
2026-10-08 22:47:42.984  | 00:00:05.299 | 159     | CRITICAL | VIRTUAL_FENCE_BREACH     | Person [ID #2]: Intruder breached Border Virtual Fence line! Vector: Inbound (Restricted Crossing) (Conf: 0.86) [BBox: 720,215,818,469]
2026-10-08 22:47:46.347  | 00:00:07.533 | 226     | INFO     | FACE_DETECTED            | Human Face [ID #3]: Facial features captured on Person #3 (Size: 26x26px) (Conf: 0.86) [BBox: 306,519,332,545]
2026-10-08 22:47:49.740  | 00:00:09.866 | 296     | WARNING  | ENVIRONMENT_CHANGE       | SYSTEM: Night-Vision mode ENGAGED (Low-Light Condition) | Mean Luminance: 64.8/255
2026-10-08 22:47:49.782  | 00:00:09.866 | 296     | WARNING  | NIGHT_TIME_MOVEMENT      | Person [ID #1]: Movement detected under low-light/night conditions at (931, 482) (Conf: 0.89) [BBox: 882,229,980,482]

==============================================================================================================
                                      SURVEILLANCE ANALYTICS SESSION SUMMARY
==============================================================================================================
Camera ID             : BOP_NORTH_SECTOR_CAM_01
Session Ended         : 2026-10-08 22:47:54.504
Total Video Frames    : 360 (00:00:12.000)
Processed Video Length: 12.00 seconds
--------------------------------------------------------------------------------------------------------------
Unique Humans Tracked : 9
Unique Vehicles       : 3
Critical Alerts       : 23
Warnings Triggered    : 6
Information Events    : 20
Total Logged Events   : 49
==============================================================================================================
```

### 2. Machine-Readable JSONL Log (`border_events.jsonl`)

```json
{"wall_time": "2026-10-08T22:47:42.984102", "video_timecode": "00:00:05.299", "frame_idx": 159, "camera_id": "BOP_NORTH_SECTOR_CAM_01", "severity": "CRITICAL", "category": "VIRTUAL_FENCE_BREACH", "entity_type": "Person", "track_id": 2, "confidence": 0.86, "bbox": [720, 215, 818, 469], "details": "Intruder breached Border Virtual Fence line! Vector: Inbound (Restricted Crossing)", "extra": {"direction": "Inbound (Restricted Crossing)", "crossing_point": [769, 469]}}
```

---

## 🧪 Synthetic Test Generator

To generate a new synthetic test video for quick verification:

```bash
python create_test_video.py
```
This produces `border_cctv_simulation.mp4` featuring an outpost background, walking intruders, crossing trajectories, loitering persons, vehicle motion with number plates, and nightfall lighting.
