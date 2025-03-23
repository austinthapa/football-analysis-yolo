# Football Analysis Using YOLO

## Description

## Project Structure
football-analysis-yolo/
|-- README.md
|-- requirements.txt
|-- .gitignore
|-- LICENSE
|-- config.py                        # Configuration settings
|-- main.py                          # Main Entry point
|-- yolo_inference.ipynb             # Notebook for inference
|-- yolov9m.pt                       # YOLO model weights
|-- football-analysis-venv/          # Virtual environment
|-- football-players-detection-dataset/ # Dataset
|-- input_videos/
    |-- input_video.mp4               # Source video
|-- output_videos/
    |-- output.avi                   # Processed Video
|--models/ # Model configurations
|--runs/ # Training runs
|--utils/
  |-- __init__.py
  |-- video_utils.pt                 # Video Processing utilties
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
