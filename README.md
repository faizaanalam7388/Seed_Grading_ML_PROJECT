# Seed Grading ML Project

An OpenCV-based seed grading and sorting prototype. The pipeline detects seed contours, classifies seeds using HSV brightness and saturation, and schedules a mock air-valve ejection for defective seeds.

## Features

- Video or webcam input through the mock camera
- HSV-based contour detection and seed classification
- Ejection timing based on a free-fall model
- Virtual valve controller for safe simulation
- Interactive HSV threshold tuning with `hsv_tuner.py`

## Requirements

Python 3 with OpenCV and NumPy:

```bash
pip install opencv-python numpy
```

## Run

```bash
python main.py
```

The default mode uses the included test video and a simulated valve. Press `q` in the preview window to stop.