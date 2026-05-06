# 🎧 Audio World — Real-Time Assistive Object Detection

> An intelligent navigation aid for the visually impaired using **YOLOv8**, spatial distance estimation, and **text-to-speech** audio feedback.

---

## 📌 Overview

**Audio World** processes live video frames in real time to detect nearby people, estimate their distance, and announce their position (LEFT / FORWARD / RIGHT) via spoken audio. It is designed to be a practical, affordable assistive tool for individuals with visual impairment navigating dynamic public environments.

Key capabilities:
- 🎯 Real-time person detection using **YOLOv8**
- 📏 Distance estimation via bounding box geometry
- 🔊 Non-blocking **text-to-speech** announcements (threaded)
- 🔒 **Privacy-preserving** head blur for detected individuals
- 📊 Per-class evaluation: **Precision, Recall, Accuracy** (IoU-based)
- 🖥️ Full-screen visual overlay with color-coded bounding boxes

---

## 🗂️ Project Structure

```
audio-world/
├── main.py                  # Entry point — CLI args, main loop
├── requirements.txt
├── utils/
│   ├── detection.py         # YOLOv8 inference, distance & position logic
│   ├── audio.py             # TTS worker thread
│   ├── evaluation.py        # IoU-based precision/recall/accuracy
│   └── visualization.py     # Bounding box drawing & window setup
└── README.md
```

---

## ⚙️ Installation

### 1. Clone the repository

```bash
git clone https://github.com/YOUR_USERNAME/audio-world.git
cd audio-world
```

### 2. Create a virtual environment (recommended)

```bash
python -m venv venv
source venv/bin/activate        # Linux / macOS
venv\Scripts\activate           # Windows
```

### 3. Install dependencies

```bash
pip install -r requirements.txt
```

> **Note (Linux/macOS):** `pyttsx3` may require additional system packages:
> ```bash
> sudo apt install espeak ffmpeg libespeak1   # Ubuntu/Debian
> brew install espeak                          # macOS
> ```

The YOLOv8 model weights (`yolov8n.pt`) are downloaded automatically on first run.

---

## 🚀 Usage

```bash
# Run on webcam (default)
python main.py --source 0

# Run on a video file
python main.py --source path/to/video.mp4

# Fullscreen display
python main.py --source 0 --fullscreen

# Disable privacy blur
python main.py --source 0 --no-blur

# Custom alert range (meters)
python main.py --source 0 --max-distance 8.0

# Use a larger YOLOv8 model for better accuracy
python main.py --source 0 --model yolov8s.pt
```

### All CLI Options

| Argument          | Default       | Description                              |
|-------------------|---------------|------------------------------------------|
| `--source`        | `0`           | Webcam index or path to video file       |
| `--model`         | `yolov8n.pt`  | YOLOv8 weights file                      |
| `--conf`          | `0.25`        | Detection confidence threshold           |
| `--iou`           | `0.4`         | NMS IoU threshold                        |
| `--max-distance`  | `12.5`        | Max alert distance in meters             |
| `--no-blur`       | *(off)*       | Disable head blurring                    |
| `--fullscreen`    | *(off)*       | Fullscreen display mode                  |
| `--speech-rate`   | `235`         | TTS speed (words per minute)             |

Press **`q`** to quit.

---

## 📊 Evaluation Metrics

At the end of each session, a per-class performance report is printed:

```
============================================================
  DETECTION EVALUATION REPORT
============================================================
  Class        Precision   Recall   Accuracy    TP    FP    FN
  --------------------------------------------------------
  person          1.0000   1.0000     1.0000   142     0     0
============================================================
```

> ⚠️ **Note:** Ground truths are currently simulated (set equal to predictions) for development testing. For real evaluation, supply annotated ground truth labels.

---

## 🛠️ Hardware & Software Requirements

### Software
| Component | Library |
|-----------|---------|
| Object Detection | `ultralytics` (YOLOv8) |
| Video Processing | `opencv-python` |
| Numerical Ops | `numpy` |
| Text-to-Speech | `pyttsx3` |
| Concurrency | `threading`, `queue` (stdlib) |

### Hardware
- **Camera:** USB webcam or Raspberry Pi Camera Module
- **Compute:** Laptop/desktop with CPU (GPU optional but faster); Raspberry Pi 4 or NVIDIA Jetson Nano for edge deployment
- **Audio Output:** Speakers or headphones
- **Optional:** Portable power bank for wearable/field use

---

## 🔮 Future Scope

- [ ] Detect additional object classes (vehicles, poles, stairs)
- [ ] Sensor fusion with ultrasonic / LiDAR sensors
- [ ] Voice command interface for hands-free control
- [ ] GPS integration for outdoor route guidance
- [ ] Object tracking across frames
- [ ] Edge deployment (Raspberry Pi / Jetson)
- [ ] Cloud logging and analytics dashboard
- [ ] Haptic feedback module via vibration motors

---

## 📄 License

This project is licensed under the [MIT License](LICENSE).

---

## 🙏 Acknowledgements

- [Ultralytics YOLOv8](https://github.com/ultralytics/ultralytics)
- [OpenCV](https://opencv.org/)
- [pyttsx3](https://pyttsx3.readthedocs.io/)
