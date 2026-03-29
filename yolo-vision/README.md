# 🎯 YOLO Vision

> Real-time object detection powered by **Ultralytics YOLO** — with a polished Streamlit UI and a FastAPI REST backend.

```
╔══════════════════════════════════════════════╗
║          YOLO  VISION                        ║
║  image · video · webcam detection engine     ║
╚══════════════════════════════════════════════╝
```

---

## Features

| Mode | Description |
|------|-------------|
| 🖼 Image | Upload any image — detect objects, view bounding boxes, download cropped regions |
| 🎬 Video | Upload a video — every frame is annotated and saved as a new video |
| 📡 Webcam | Open your camera — live YOLO detection streamed in real time |
| 🌐 API | FastAPI backend with `/detect/image`, `/detect/video`, `/stream/webcam` endpoints |
| 💻 CLI | `detect_cli.py` for headless / scripting use cases |

Each detection result includes:
- **Class label** (e.g. `person`, `car`, `dog`)
- **Confidence score** (0 – 1)
- **Bounding box coordinates** (x1, y1, x2, y2)

---

## Project Structure

```
yolo-vision/
├── app/
│   ├── core/
│   │   ├── __init__.py
│   │   └── detector.py          # YOLODetector + Detection dataclass
│   ├── api/
│   │   ├── __init__.py
│   │   └── routes.py            # FastAPI routes
│   ├── ui/
│   │   ├── __init__.py
│   │   └── streamlit_app.py     # Streamlit UI
│   └── utils/
│       ├── __init__.py
│       └── helpers.py           # Encoding, path, summary utilities
├── outputs/
│   ├── images/                  # Annotated image outputs
│   ├── videos/                  # Annotated video outputs
│   └── crops/                   # Cropped detection regions
├── tests/
│   └── test_core.py             # Unit tests (pytest)
├── detect_cli.py                # CLI entry point
├── main_api.py                  # FastAPI server entry point
├── requirements.txt
└── README.md
```

---

## Setup

### 1. Clone the repo

```bash
git clone https://github.com/yourname/yolo-vision.git
cd yolo-vision
```

### 2. Create a virtual environment

```bash
python -m venv .venv
source .venv/bin/activate          # Windows: .venv\Scripts\activate
```

### 3. Install dependencies

```bash
pip install -r requirements.txt
```

> **Note**: `ultralytics` will automatically download PyTorch if it's not installed.

### 4. Get a YOLO model

You can use any Ultralytics model. The easiest way is to let Ultralytics auto-download:

```python
# Models auto-download on first use
# Available sizes: yolov8n.pt  yolov8s.pt  yolov8m.pt  yolov8l.pt  yolov8x.pt
#                  yolo11n.pt  yolo11s.pt  yolo11m.pt  ...
```

Or download manually from [Ultralytics releases](https://github.com/ultralytics/ultralytics/releases).

---

## How to Run

### Option A — Streamlit UI (recommended)

```bash
streamlit run app/ui/streamlit_app.py
```

Open [http://localhost:8501](http://localhost:8501) in your browser.

1. Enter your model path in the **sidebar** (e.g. `yolov8n.pt`)
2. Click **LOAD MODEL**
3. Choose a tab: **Image**, **Video**, or **Webcam**

---

### Option B — FastAPI REST API

```bash
# Set model path via environment variable (optional)
export YOLO_MODEL_PATH=yolov8n.pt
export CONF_THRESHOLD=0.35

# Start the server
python main_api.py
# or
uvicorn main_api:app --host 0.0.0.0 --port 8000 --reload
```

Interactive docs at [http://localhost:8000/docs](http://localhost:8000/docs)

**Key endpoints:**

```
GET  /health                     → model status and class list
POST /detect/image               → upload image, get detection JSON + annotated URL
POST /detect/video               → upload video, get download URL for annotated video
GET  /stream/webcam?camera=0     → MJPEG webcam stream
PATCH /config?confidence=0.4     → update thresholds at runtime
```

---

### Option C — CLI

```bash
# Image detection
python detect_cli.py image --model yolov8n.pt --input photo.jpg --show

# Video detection
python detect_cli.py video --model yolov8n.pt --input clip.mp4 --output result.mp4

# Webcam (press Q to quit)
python detect_cli.py webcam --model yolov8n.pt --camera 0

# Extra flags
python detect_cli.py image --model yolov8n.pt --input photo.jpg --conf 0.4 --iou 0.5
```

---

## Run Tests

```bash
pytest tests/ -v
```

---

## Environment Variables (API mode)

| Variable | Default | Description |
|----------|---------|-------------|
| `YOLO_MODEL_PATH` | `yolov8n.pt` | Path to the YOLO weights file |
| `CONF_THRESHOLD` | `0.25` | Detection confidence threshold |
| `IOU_THRESHOLD` | `0.45` | NMS IOU threshold |
| `API_HOST` | `0.0.0.0` | FastAPI bind host |
| `API_PORT` | `8000` | FastAPI bind port |

---

## Output Files

All outputs are saved under the `outputs/` directory:

```
outputs/
├── images/     # Annotated images  (JPG)
├── videos/     # Annotated videos  (MP4)
└── crops/      # Cropped objects   (JPG, one per detection)
```

In API mode, files are served statically at `/outputs/...`.

---

## Tech Stack

- **[Ultralytics YOLO](https://github.com/ultralytics/ultralytics)** — detection backbone
- **[OpenCV](https://opencv.org/)** — image/video I/O and annotation drawing
- **[Streamlit](https://streamlit.io/)** — interactive web UI
- **[FastAPI](https://fastapi.tiangolo.com/)** — REST API server
- **[Pillow](https://python-pillow.org/)** — image format conversion
- **[NumPy](https://numpy.org/)** — array processing

---

## Notes

- **Webcam** requires a physical camera. In headless / server environments the webcam tab won't work, but the image and video modes will.
- For GPU acceleration, install the CUDA-enabled version of PyTorch *before* installing `ultralytics`.
- The Streamlit UI caches the detector in `st.session_state` to avoid reloading between tab switches.

---

*Built as a professional portfolio project — modular, documented, and ready to extend.*
