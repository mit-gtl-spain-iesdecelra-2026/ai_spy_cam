#!/usr/bin/env python3
"""
Lightweight YOLO Object Detection for Raspberry Pi
===================================================
Optimized for Raspberry Pi 4 + Camera Module 3 + 7" Display

For teaching students about real-time AI object detection.

Controls:
- Press 'q' to quit
- Press 's' to save a screenshot
- Press 'p' to pause/resume detection

Author: AI Spy Cam Project
"""

import cv2
import time
import subprocess
import numpy as np
from pathlib import Path
from datetime import datetime

# Import YOLO from ultralytics
from ultralytics import YOLO


# =============================================================================
# CONFIGURATION - Easy to modify for teaching
# =============================================================================

# Model selection (yolov8n = nano = fastest, smallest)
# Options: yolov8n, yolov8s, yolov8m, yolov8l, yolov8x (n=nano is best for Pi)
MODEL_NAME = "yolov8n.pt"

# Camera resolution (lower = faster, 640x480 is good balance)
CAMERA_WIDTH = 640
CAMERA_HEIGHT = 480

# Display settings (matched to 7" screen at 800x480)
DISPLAY_WIDTH = 800
DISPLAY_HEIGHT = 480

# Detection settings
CONFIDENCE_THRESHOLD = 0.5  # Only show detections above this confidence (0-1)
IOU_THRESHOLD = 0.45        # Intersection over Union threshold for NMS

# Performance settings
INFERENCE_SIZE = 320        # Input size for YOLO (320 is fast, 640 is accurate)

# Visual settings
BOX_THICKNESS = 2
FONT_SCALE = 0.6
FONT_THICKNESS = 2

# Colors (BGR format for OpenCV)
COLORS = [
    (255, 100, 100),   # Light blue
    (100, 255, 100),   # Light green
    (100, 100, 255),   # Light red
    (255, 255, 100),   # Cyan
    (255, 100, 255),   # Magenta
    (100, 255, 255),   # Yellow
    (200, 150, 100),   # Steel blue
    (100, 200, 150),   # Sea green
]


def get_color(class_id):
    """Get a consistent color for each class ID."""
    return COLORS[class_id % len(COLORS)]


def draw_detections(frame, results):
    """Draw bounding boxes and labels on the frame."""
    if results[0].boxes is None or len(results[0].boxes) == 0:
        return frame, 0
    
    boxes = results[0].boxes
    count = 0
    
    for box in boxes:
        # Get box coordinates
        x1, y1, x2, y2 = map(int, box.xyxy[0])
        
        # Get confidence and class
        conf = float(box.conf[0])
        class_id = int(box.cls[0])
        class_name = results[0].names[class_id]
        
        # Get color for this class
        color = get_color(class_id)
        
        # Draw bounding box
        cv2.rectangle(frame, (x1, y1), (x2, y2), color, BOX_THICKNESS)
        
        # Create label with class name and confidence
        label = f"{class_name}: {conf:.0%}"
        
        # Calculate label size for background
        (label_w, label_h), baseline = cv2.getTextSize(
            label, cv2.FONT_HERSHEY_SIMPLEX, FONT_SCALE, FONT_THICKNESS
        )
        
        # Draw label background
        cv2.rectangle(
            frame,
            (x1, y1 - label_h - 10),
            (x1 + label_w + 5, y1),
            color,
            -1
        )
        
        # Draw label text (white on colored background)
        cv2.putText(
            frame,
            label,
            (x1 + 2, y1 - 5),
            cv2.FONT_HERSHEY_SIMPLEX,
            FONT_SCALE,
            (255, 255, 255),
            FONT_THICKNESS
        )
        
        count += 1
    
    return frame, count


def draw_info_overlay(frame, fps, detection_count, paused=False):
    """Draw performance info overlay on the frame."""
    h, w = frame.shape[:2]
    
    # Semi-transparent background for info
    overlay = frame.copy()
    cv2.rectangle(overlay, (10, 10), (220, 90), (0, 0, 0), -1)
    cv2.addWeighted(overlay, 0.6, frame, 0.4, 0, frame)
    
    # Draw info text
    cv2.putText(frame, f"FPS: {fps:.1f}", (20, 35),
                cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2)
    cv2.putText(frame, f"Objects: {detection_count}", (20, 60),
                cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 255), 2)
    cv2.putText(frame, f"Model: {MODEL_NAME}", (20, 85),
                cv2.FONT_HERSHEY_SIMPLEX, 0.5, (200, 200, 200), 1)
    
    if paused:
        # Draw PAUSED indicator
        cv2.putText(frame, "PAUSED", (w//2 - 60, 40),
                    cv2.FONT_HERSHEY_SIMPLEX, 1.0, (0, 0, 255), 3)
    
    # Draw controls hint at bottom
    cv2.putText(frame, "Q=Quit  S=Save  P=Pause", (w//2 - 120, h - 15),
                cv2.FONT_HERSHEY_SIMPLEX, 0.5, (180, 180, 180), 1)
    
    return frame


def save_screenshot(frame):
    """Save current frame as a screenshot."""
    screenshots_dir = Path("/home/pi/ai_spy_cam/screenshots")
    screenshots_dir.mkdir(exist_ok=True)
    
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    filename = screenshots_dir / f"detection_{timestamp}.jpg"
    cv2.imwrite(str(filename), frame)
    print(f"Screenshot saved: {filename}")
    return filename


def init_camera():
    """
    Initialize camera using rpicam-vid for Pi Camera Module 3.
    This uses the modern libcamera stack via rpicam-vid subprocess.
    """
    # Use rpicam-vid to output raw video frames that OpenCV can read
    # This is the most reliable method for Pi Camera Module 3
    print("   Using rpicam-vid (libcamera backend)...")
    
    # rpicam-vid command that outputs to stdout
    # --inline: include headers in stream
    # --flush: flush output immediately (reduces latency)
    # -t 0: run indefinitely
    rpicam_cmd = [
        "rpicam-vid",
        "-t", "0",                          # Run forever
        "--width", str(CAMERA_WIDTH),
        "--height", str(CAMERA_HEIGHT),
        "--framerate", "30",
        "--codec", "yuv420",                # Raw YUV output
        "--flush",                          # Low latency
        "-o", "-"                           # Output to stdout
    ]
    
    try:
        # Start rpicam-vid process
        process = subprocess.Popen(
            rpicam_cmd,
            stdout=subprocess.PIPE,
            stderr=subprocess.DEVNULL,
            bufsize=CAMERA_WIDTH * CAMERA_HEIGHT * 3 // 2  # YUV420 buffer
        )
        
        # Calculate frame size for YUV420
        frame_size = CAMERA_WIDTH * CAMERA_HEIGHT * 3 // 2
        
        print("   [OK] rpicam-vid started!")
        return process, "rpicam", frame_size
        
    except FileNotFoundError:
        print("   [WARN] rpicam-vid not found, trying fallback...")
    except Exception as e:
        print(f"   [WARN] rpicam-vid failed: {e}")
    
    # Fallback: Try OpenCV directly (for USB cameras or testing)
    print("   Trying OpenCV fallback...")
    cap = cv2.VideoCapture(0)
    if cap.isOpened():
        cap.set(cv2.CAP_PROP_FRAME_WIDTH, CAMERA_WIDTH)
        cap.set(cv2.CAP_PROP_FRAME_HEIGHT, CAMERA_HEIGHT)
        cap.set(cv2.CAP_PROP_BUFFERSIZE, 1)
        print("   [OK] OpenCV capture working!")
        return cap, "opencv", 0
    
    return None, None, 0


def read_frame_rpicam(process, frame_size):
    """Read a single frame from rpicam-vid process."""
    # Read YUV420 data
    raw_data = process.stdout.read(frame_size)
    if len(raw_data) != frame_size:
        return None
    
    # Convert YUV420 to numpy array
    yuv = np.frombuffer(raw_data, dtype=np.uint8).reshape(
        (CAMERA_HEIGHT * 3 // 2, CAMERA_WIDTH)
    )
    
    # Convert YUV420 to BGR for OpenCV
    frame = cv2.cvtColor(yuv, cv2.COLOR_YUV2BGR_I420)
    return frame


def main():
    """Main detection loop."""
    print("=" * 60)
    print("YOLO Object Detection - Raspberry Pi")
    print("=" * 60)
    print(f"Camera: {CAMERA_WIDTH}x{CAMERA_HEIGHT}")
    print(f"Display: {DISPLAY_WIDTH}x{DISPLAY_HEIGHT}")
    print(f"Model: {MODEL_NAME}")
    print(f"Inference size: {INFERENCE_SIZE}px")
    print("=" * 60)
    
    # Initialize the camera
    print("\nInitializing camera...")
    camera, method, frame_size = init_camera()
    
    if camera is None:
        print("[ERROR] Could not initialize camera!")
        print("   Make sure the camera is connected and enabled.")
        print("   Try: rpicam-hello --list-cameras")
        return
    
    # Let camera warm up
    time.sleep(0.5)
    print(f"[OK] Camera ready! (using {method})")
    
    # Load YOLO model
    print(f"\nLoading {MODEL_NAME}...")
    model = YOLO(MODEL_NAME)
    print("[OK] Model loaded!")
    
    # Create window
    window_name = "YOLO Detection - Raspberry Pi"
    cv2.namedWindow(window_name, cv2.WINDOW_NORMAL)
    cv2.resizeWindow(window_name, DISPLAY_WIDTH, DISPLAY_HEIGHT)
    cv2.moveWindow(window_name, 0, 0)  # Position at top-left
    
    print("\nStarting detection...")
    print("   Press 'q' to quit, 's' to save screenshot, 'p' to pause\n")
    
    # Performance tracking
    fps = 0
    frame_count = 0
    start_time = time.time()
    paused = False
    last_frame = None
    last_results = None
    
    try:
        while True:
            # Capture frame based on method
            if method == "rpicam":
                frame = read_frame_rpicam(camera, frame_size)
            else:
                ret, frame = camera.read()
                if not ret:
                    frame = None
            
            if frame is None:
                print("[WARN] Failed to capture frame, retrying...")
                time.sleep(0.1)
                continue
            
            if not paused:
                # Run YOLO inference
                results = model.predict(
                    frame,
                    imgsz=INFERENCE_SIZE,
                    conf=CONFIDENCE_THRESHOLD,
                    iou=IOU_THRESHOLD,
                    verbose=False,
                    device='cpu'  # Use CPU on Raspberry Pi
                )
                
                # Draw detections
                frame, detection_count = draw_detections(frame, results)
                last_frame = frame.copy()
                last_results = results
            else:
                # When paused, use last results
                if last_results is not None:
                    frame, detection_count = draw_detections(frame, last_results)
                else:
                    detection_count = 0
            
            # Resize frame to fit display
            if frame.shape[1] != DISPLAY_WIDTH or frame.shape[0] != DISPLAY_HEIGHT:
                frame = cv2.resize(frame, (DISPLAY_WIDTH, DISPLAY_HEIGHT))
            
            # Calculate FPS
            frame_count += 1
            elapsed = time.time() - start_time
            if elapsed > 0.5:  # Update FPS every 0.5 seconds
                fps = frame_count / elapsed
                frame_count = 0
                start_time = time.time()
            
            # Draw info overlay
            frame = draw_info_overlay(frame, fps, detection_count, paused)
            
            # Display frame
            cv2.imshow(window_name, frame)
            
            # Handle keyboard input
            key = cv2.waitKey(1) & 0xFF
            
            if key == ord('q'):
                print("\nQuitting...")
                break
            elif key == ord('s'):
                save_screenshot(frame)
            elif key == ord('p'):
                paused = not paused
                status = "PAUSED" if paused else "RESUMED"
                print(f"Detection {status}")
    
    except KeyboardInterrupt:
        print("\n\nInterrupted by user")
    
    finally:
        # Cleanup
        print("Cleaning up...")
        if method == "rpicam":
            camera.terminate()
            camera.wait()
        else:
            camera.release()
        cv2.destroyAllWindows()
        print("Done!")


if __name__ == "__main__":
    main()
