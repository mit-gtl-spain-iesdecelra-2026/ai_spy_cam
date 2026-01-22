# AI Spy Cam - YOLO Object Detection for Raspberry Pi

A lightweight real-time object detection system using YOLOv8 on Raspberry Pi.  
**Perfect for teaching students about AI and computer vision!**

![Python](https://img.shields.io/badge/Python-3.11+-blue)
![Platform](https://img.shields.io/badge/Platform-Raspberry%20Pi%204-red)
![License](https://img.shields.io/badge/License-MIT-green)

---

## Hardware Requirements

| Component | Tested Model |
|-----------|--------------|
| Computer | Raspberry Pi 4 Model B (4GB+ recommended) |
| Camera | Pi Camera Module 3 |
| Display | Official 7" Touchscreen (800x480) |
| Storage | 16GB+ microSD card |

## Quick Start

### 1. Clone the Repository

```bash
git clone https://github.com/mit-gtl-spain-iesdecelra-2026/ai_spy_cam.git
cd ai_spy_cam
```

### 2. Create Virtual Environment

```bash
python3 -m venv ai
source ai/bin/activate
```

### 3. Install System Dependencies

```bash
sudo apt update
sudo apt install -y libcap-dev python3-libcamera python3-picamera2
```

### 4. Install Python Dependencies

```bash
pip install -r requirements.txt
```

### 5. Run Detection

```bash
DISPLAY=:0 python3 yolo_detect.py
```

> **Note:** The `DISPLAY=:0` is needed when running via SSH to display on the Pi's screen.

---

## Controls

| Key | Action |
|-----|--------|
| `Q` | Quit the application |
| `S` | Save screenshot to `screenshots/` folder |
| `P` | Pause/Resume detection |

---

## Configuration

Edit the top of `yolo_detect.py` to customize:

```python
# Model selection (yolov8n = nano = fastest for Pi)
MODEL_NAME = "yolov8n.pt"

# Camera resolution (lower = faster)
CAMERA_WIDTH = 640
CAMERA_HEIGHT = 480

# Detection confidence threshold (0-1)
CONFIDENCE_THRESHOLD = 0.5

# Inference size (320 is fast, 640 is accurate)
INFERENCE_SIZE = 320
```

### Model Options

| Model | Size | Speed | Accuracy |
|-------|------|-------|----------|
| `yolov8n.pt` | 6MB | Fastest | Good |
| `yolov8s.pt` | 22MB | Fast | Better |
| `yolov8m.pt` | 52MB | Medium | Great |

> **Recommended:** Use `yolov8n.pt` (nano) for Raspberry Pi 4

---

## Troubleshooting

### Camera not detected

```bash
# Check if camera is recognized
rpicam-hello --list-cameras
```

### Display not showing

```bash
# Make sure you're using the correct display
echo $DISPLAY  # Should show :0

# If running via SSH, set display explicitly
export DISPLAY=:0
```

### Slow performance

1. Use the nano model (`yolov8n.pt`)
2. Lower `INFERENCE_SIZE` to 320
3. Reduce camera resolution to 480x360

---

## For Students

### What is YOLO?

**YOLO** stands for "You Only Look Once" - it's a real-time object detection algorithm that can identify multiple objects in images and video in a single pass.

### How does it work?

1. **Capture** - Camera captures a frame
2. **Process** - Image is resized and normalized
3. **Detect** - YOLO neural network analyzes the image
4. **Output** - Bounding boxes and labels are drawn

### Objects it can detect

YOLOv8 is trained on the COCO dataset with 80 classes including:
- People, bicycles, cars, motorcycles
- Dogs, cats, birds, horses
- Chairs, couches, TVs, laptops
- And many more!

---

## Project Structure

```
ai_spy_cam/
├── yolo_detect.py      # Main detection script
├── requirements.txt    # Python dependencies
├── README.md          # This file
├── .gitignore         # Git ignore rules
└── screenshots/       # Saved screenshots (created at runtime)
```

---

## Credits

- [Ultralytics YOLOv8](https://github.com/ultralytics/ultralytics) - Object detection model
- [OpenCV](https://opencv.org/) - Computer vision library
- [Raspberry Pi Foundation](https://www.raspberrypi.org/) - Hardware platform

---

## License

MIT License - Feel free to use this for educational purposes!
