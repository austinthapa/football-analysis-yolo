# Football Analysis Using YOLO

## Description
This project leverages YOLOv9, OpenCV, and tracking algorithms to detect football players, assign them to their respective teams based on jersey color, and generate processed output videos. It includes video processing, player tracking, and color-based team assignment – all wrapped in a modular, production-ready structure.

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

##  Features

- Fine-tuned YOLOv11 for football player detection
- Real-time player tracking and ID consistency across frames
- Automatic team assignment via dominant color detection
- Output video generation with overlays and team labels

##  Setup

```bash
git clone https://github.com/austinthapa/football-analysis-yolo.git
cd football-analysis-yolo
pip install -r requirements.txt
```
## Authors
Austin Thapa

## License
MIT License
