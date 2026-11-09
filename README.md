# Neural CCTV Analyser
### Smart Everyday Household CCTV & Automated Perimeter Monitor

**Neural CCTV Analyser** transforms conventional IP-based household security cameras into an intelligent, autonomous home guardian. Unlike traditional CCTV systems that continuously burn high GPU compute or send private video streams to the cloud, **Neural CCTV Analyser runs 100% locally and offline** with zero cloud subscription fees.

---

## 🌟 Key Capabilities

| Capability | Module & Implementation | Output in Log |
|---|---|---|
| **10-Frame Anchor Motion Gate** | CPU-level temporal absolute differencing ($F_k - F_0$) across 10-frame chunks | Skips 90%+ of idle video, keeping GPU cool & power consumption minimal |
| **Virtual Fencing & Household Zones** | Polygonal geo-fencing for `doorstep`, `driveway`, and `porch walkway` | Pinpoints exactly where people and vehicles are located |
| **Local 512-d Face Vector Recognition** | Inception-ResNet-V1 + MTCNN landmark alignment | Converts face crops into 512-dimensional latent vectors |
| **Local Vector Database Storage** | ChromaDB persistent store with cosine similarity matching | Matches family members (`Father`, `Mother`) in $<1$ ms locally |
| **Unknown Face Triage & Tagging** | Automatic snapshot capture + interactive CLI tagging (`tag_face.py`) | Flags unknown persons, saves snapshots, and allows one-command naming |
| **Semantic Natural Language Activity Logging** | Spatial-temporal rule synthesis + object co-occurrence | Generates human sentences like: *"Father is home. Arrived at your doorstep."* or *"A person was on your door carrying a package."* |
| **Package & Parcel Delivery Detection** | COCO parcel classes + torso contour detection | Flags couriers and delivery packages at the doorstep |
| **Visual Security HUD Export** | OpenCV canvas rendering with zone boundaries, bounding boxes, and activity banner | Generates annotated video (`.mp4`) with real-time security HUD |

---

## 📁 Project Architecture

```
Neural-CCTV-Analyser/
├── neural_cctv/
│   ├── __init__.py
│   ├── motion_gate.py         # 10-frame anchor temporal differencing (zero-GPU gatekeeper)
│   ├── face_engine.py         # 512-d Inception-ResNet embedding & local ChromaDB store
│   └── semantic_engine.py     # Household zones & natural language activity synthesizer
├── data/
│   ├── face_db/               # Local persistent ChromaDB vector database
│   └── unknown_faces/         # Captured snapshots of unknown visitors awaiting user naming
├── config.json                # Household zones, motion thresholds, and camera configuration
├── neural_analyzer.py         # Main surveillance CLI entrypoint
├── tag_face.py                # Face enrollment and tagging CLI tool
├── create_household_video.py  # Synthetic household CCTV test footage generator
├── requirements.txt           # Open-source dependencies
└── README.md                  # System documentation
```

---

## ⚡ How the 10-Frame Anchor Motion Gate Works

Rather than executing heavy neural inference on all 30 frames per second:
1. Video is split into **10-frame chunks** ($1/3$ second slices).
2. **Frame 0** acts as the **Anchor Frame** ($F_0$).
3. The CPU computes the mean absolute difference:
   $$\bar{\Delta} = \frac{1}{9}\sum_{k=1}^{9} \text{mean}(|F_k - F_0|)$$
4. If $\bar{\Delta} < \text{Threshold}$ (e.g. $0.70\text{ px}$):
   - **Scene is Idle.** GPU inference is completely bypassed, reducing thermal load and power consumption by **50% to 95%**!
5. If $\bar{\Delta} \ge \text{Threshold}$:
   - **Activity Detected!** The neural pipeline (YOLO + Face Recognition + Semantics) is instantly awakened.

---

## 🚀 Quickstart & Usage

### 1. Ingest Video or RTSP Stream

To analyze any camera feed or recorded footage:

```bash
py neural_analyzer.py --video household_cctv_simulation.mp4 --save-video output_annotated.mp4
```

### 2. Full CLI Options

```bash
py neural_analyzer.py \
    --video path/to/cctv_stream.mp4 \
    --config config.json \
    --output-log cctv_activity.log \
    --json-log cctv_activity.jsonl \
    --save-video annotated_cctv.mp4 \
    --device auto
```

| Flag | Default | Description |
|---|---|---|
| `--video`, `-v` | *(Required)* | Input video file path (`.mp4`, `.avi`) or camera RTSP stream (`rtsp://...`) |
| `--config`, `-c` | `config.json` | Configuration file containing zones and thresholds |
| `--output-log`, `-o` | `cctv_activity.log` | Human-readable audit log file destination |
| `--json-log`, `-j` | `cctv_activity.jsonl` | Machine-readable JSONL structured event log |
| `--save-video`, `-s` | `None` | Optional path to export annotated MP4 video |
| `--display`, `-d` | `False` | Display live preview window with real-time HUD overlays |
| `--device` | `auto` | Execution device (`cuda` for GPU or `cpu`) |

---

## 👤 Face Recognition & User Tagging Workflow

### Step 1: When an Unknown Person Appears
The system captures their face, saves a high-resolution snapshot to `./data/unknown_faces/`, and logs:
```
[00:00:08.033] [NOTICE] An unknown person entered your doorstep.
```

### Step 2: List Unidentified Faces
To see pending visitors who need a name:
```bash
py tag_face.py --unknowns
```

### Step 3: Name the Person (Enroll into ChromaDB)
Assign a name to the snapshot:
```bash
py tag_face.py --tag unknown_20261029_035507_track1.jpg --name "Father"
```

### Step 4: Next Time They Appear
The system extracts their 512-d vector, performs a cosine search against ChromaDB ($<1\text{ ms}$), and automatically logs:
```
[00:00:03.500] [INFO] Father is home. Arrived at your doorstep.
```

### Step 5: Direct Photo Enrollment
You can also directly enroll any family member from an existing photo on your computer:
```bash
py tag_face.py --enroll path/to/father_photo.jpg --name "Father"
```

---

## ⚙️ Household Zones Configuration (`config.json`)

You can define any number of custom zones to match your camera angle:

```json
{
  "project_name": "Neural CCTV Analyser",
  "camera": {
    "camera_id": "FRONT_PORCH_CAM_01",
    "location": "Front Doorstep & Driveway"
  },
  "motion_gate": {
    "enabled": true,
    "chunk_size": 10,
    "motion_threshold": 0.70
  },
  "zones": [
    {
      "zone_id": "ZONE_DOORSTEP",
      "name": "doorstep",
      "polygon": [[0.42, 0.42], [0.82, 0.42], [0.82, 0.90], [0.42, 0.90]],
      "alert_level": "NOTICE"
    },
    {
      "zone_id": "ZONE_DRIVEWAY",
      "name": "driveway",
      "polygon": [[0.05, 0.35], [0.40, 0.35], [0.40, 0.95], [0.05, 0.95]],
      "alert_level": "INFO"
    }
  ],
  "face_recognition": {
    "enabled": true,
    "distance_threshold": 0.38,
    "db_path": "./data/face_db",
    "unknown_dir": "./data/unknown_faces"
  }
}
```

---

## 🧪 Testing with Synthetic Household Footage

Generate a synthetic 18-second household video featuring static idle periods, vehicle arrivals, couriers carrying packages, and doorstep entries:

```bash
py create_household_video.py
```
This produces `household_cctv_simulation.mp4` for immediate testing.
