# Football Analysis Using YOLO

## Description

## Project Structure

```
.
├── LICENSE
├── README.md                   # Read me
├── analysis                    # Notebook to process
│   └── player_team_color_assignment.ipynb
├── config.py                   # Store API Key
├── documentation.txt           # Documentation for
├── input_videos
│   └── input_video.mp4         #
├── main.py                     # Entry Point
├── models                      # Best weights from trained YOLOv9 model
│   └── best.pt
├── output_videos               # Store the processed video frames
│   ├── cropped_img.jpg
│   └── output.avi
├── requirements.txt            # Dependencies and Requirements
├── stubs
│   └── tracks_stub_long.pkl
├── team_assigner               # Assign player to a team
│   ├── __init__.py
│   └── team_assigner.py
├── trackers                    # Run Tracker on each frame for players
│   ├── __init__.py
│   └── video_tracker.py
├── utils                       # Utilities for bounding box operations
│   ├── __init__.py
│   ├── bbox_utils.py           #
│   └── video_utils.py          #
├── yolo_inference_final.ipynb  # Fine-tune YOLO model on subset of Football image datasets
└── yolov9m.pt                  # Original YOLOv9 medium model
```

## Getting Started

### Dependencies

### Installing

## Help

## Authors

## License

## Acknowledegments

Football Analysis Project using OpenCV, YOLO

numpy
pandas
scikit-learn

opencv-python
ultralytics
roboflow
supervision

ultralytics require numpy<=2.1.1 and numpy>=1.23.0
ERROR: pip's dependency resolver does not currently take into account all the packages that are installed. This behaviour is the source of the following dependency conflicts.
ultralytics 8.3.94 requires numpy<=2.1.1,>=1.23.0, but you have numpy 2.2.4 which is incompatible.

YOLOv8 vs YOLOv9: Which one to choose for football detection?
YOLOv8

- It has specific improvements for small object detection (useful for distant players and the ball)

YOLOv9
