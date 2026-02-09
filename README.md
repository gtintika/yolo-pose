# Pose Detection with Duplicate Removal

YOLO26 pose detection with automatic duplicate person filtering and motion analysis.

## Installation

At Linux / macOS terminal application, run:

```bash

# choose project location
mkdir -p ~/projects;cd ~/projects
# clone repository
git clone https://github.com/gtintika/yolo-pose.git

cd yolo-pose

# prepare uv environment
uv sync
source .venv/bin/activate

# or 

uv sync
uv run python --help

```

## Models

```bash
# you can use any YOLOv8 pose model, but the large version is recommended for best accuracy
wget https://github.com/ultralytics/assets/releases/download/v8.4.0/yolo26l-pose.pt

# default model YOLO26m-pose.pt

# more details at:

https://docs.ultralytics.com/tasks/pose/
```

## Usage

```bash
# Basic pose detection
python pose_detection.py --source video.mp4

# With motion analysis (headless, save output)
python pose_detection.py --source abduction_2.mp4 --output abduction_2_annotated.mp4 --no-show \
--analyze '{right_arm_abduction,left_arm_abduction,right_elbow_flexion,left_elbow_flexion,right_knee_flexion,left_knee_flexion}'

# Use specific device
python pose_detection.py --source video.mp4 --device cpu

# Webcam
python pose_detection.py --source 0
```




## CLI Reference

| Flag | Type | Default | Description |
|------|------|---------|-------------|
| `--source` | str | **required** | Video file path or camera index |
| `--model` | str | `yolo26m-pose.pt` | YOLO model name |
| `--conf` | float | `0.5` | Detection confidence threshold |
| `--output` | str | None | Output video path |
| `--no-show` | flag | off | Do not display video window |
| `--analyze` | str | `''` | Motions to analyze (see below) |
| `--device` | str | auto | `cpu`, `mps`, `cuda`, `cuda:0` |
| `--keep-duplicates` | flag | off | Disable duplicate filtering |
| `--duplicate-threshold` | float | `0.7` | Keypoint similarity threshold (0-1, lower = more aggressive) |

## Motion Analysis

Pass motion types to `--analyze` as comma-separated values in braces:

```bash
--analyze '{right_arm_abduction,left_knee_flexion}'
```

Available motions:

| Motion | Description |
|--------|-------------|
| `right_arm_abduction` | Right arm raising to the side (shoulder-wrist vs vertical) |
| `left_arm_abduction` | Left arm raising to the side |
| `right_elbow_flexion` | Right elbow bending |
| `left_elbow_flexion` | Left elbow bending |
| `right_knee_flexion` | Right knee bending |
| `left_knee_flexion` | Left knee bending |

Abduction includes a lateral check — the arm must extend outward (not cross the body) to register.

Each motion outputs a normalized value (0.0-1.0) and the raw angle in degrees.

## Stats Output

After processing, per-run stats are printed:

```
--- Stats ---
Frames processed: 120
Avg inference:    45.2 ms/frame
Avg processing:   48.1 ms/frame (inference + annotation + I/O)
Throughput:       20.8 FPS
Total time:       5.77 s
```

## Duplicate Removal

YOLO sometimes detects the same person multiple times. This is filtered by default using keypoint similarity.

- **`--duplicate-threshold 0.7`** (default): Balanced
- **Higher (0.85-0.95)**: More strict, keeps more detections (use when people stand close)
- **Lower (0.5-0.6)**: More aggressive, removes more duplicates
- **`--keep-duplicates`**: Disable filtering entirely

## Global Constants

| Constant | Default | Description |
|----------|---------|-------------|
| `DEBUG` | `False` | Enable detailed per-frame debug prints |
| `KEYPOINT_CONF_THRESHOLD` | `0.5` | Minimum keypoint confidence for motion analysis |

## Programmatic Usage

```python
from pose_detection import CleanPoseDetector

detector = CleanPoseDetector(
    model_name='yolo26m-pose.pt',
    conf_threshold=0.5,
    remove_duplicates=True,
    duplicate_threshold=0.7,
    device='mps'
)

import cv2
frame = cv2.imread('image.jpg')
annotated_frame, keypoints_list = detector.detect_pose(frame)
print(f"Detected {len(keypoints_list)} unique person(s)")
```

## References

- [Ultralytics YOLO Pose Documentation](https://docs.ultralytics.com/tasks/pose/)
- [YOLO github examples](https://github.com/ultralytics/ultralytics/tree/main/examples)
- [YOLOv8-Action-Recognition](https://github.com/ultralytics/ultralytics/tree/main/examples/YOLOv8-Action-Recognition)
