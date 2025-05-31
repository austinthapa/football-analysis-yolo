# Football Analysis Using YOLO

## Description
This project presents a modular, end-to-end computer vision pipeline for analyzing football videos using YOLOv11, OpenCV, and advanced tracking algorithms. The system performs player detection, draw bounding boxes around players, assigns player teams using jersey color clustering, estimates player speed and distance, and maps player movements onto a static top-down view of the football field through perspective transformation. We also performed camera movement estimation by tracking the features from the frames. Overall, we produced processed output videos with rich visualizations, enabling insights into match dynamics, all built in a clean, well-defined structure

---

## Demo
![Demo of Player Tracking](output.gif)

---

## Project Structure
```
.
├── LICENSE
├── README.md                   # Documentation
├── analysis                    # Notebook to get player jersey color
│   └── player_team_color_assignment.ipynb
├── config.py                   # Store API Key for RoboFlow API
├── input_videos
│   └── input_video.mp4         
├── main.py                     # Entry Point
├── models                      # Best weights from trained YOLOv11 model
│   └── best.pt
├── output_videos               # Store the processed video frames & pictures
│   ├── cropped_img.jpg
│   └── output.avi
├── requirements.txt            # Dependencies and Requirements
├── stubs
│   └── tracks_stub_long.pkl    # Store trackers to avoid re-running everytime
├── team_assigner               # Assign player to a team
│   ├── __init__.py
│   └── team_assigner.py
├── trackers                    # Run Tracker on each frame for players
│   ├── __init__.py
│   └── video_tracker.py
├── utils                       # Utilities for bounding box operations
│   ├── __init__.py
│   ├── bbox_utils.py           
│   └── video_utils.py          
├── yolo_inference_final.ipynb  # Fine-tune YOLO model on subset of Football image datasets
└── yolov11.pt                  # Original YOLOv11 medium model
```
---

## Features

-  **Object Detection**: Uses fine-tuned YOLOv11 to detect players, referees, and the ball.
-  **Tracking**: Assigns consistent IDs across frames using supervised trackers.
-  **Team Identification**: Uses k-means clustering to auto-assign players to teams based on jersey colors.
-  **Perspective Transform**: Converts the dynamic camera view to a static top-down view.
-  **Player Metrics**: Calculates real-time speed, distance, and movement patterns.
-  **Model Evaluation**: Evaluated using mAP@0.5 and mAP@0.5:0.95 metrics.

---

## Technologies Used

- [YOLOv11]
- [OpenCV](https://opencv.org/)
- [Python](https://www.python.org/)
- [Roboflow](https://roboflow.com/)
- [Scikit-learn (k-means clustering)]
- [NumPy](https://numpy.org/)

---

## Setup Instructions

1. **Clone the Repository**

```bash
git clone https://github.com/austinthapa/football-analysis-yolo.git
cd football-analysis-yolo
````

2. **Create a Virtual Environment**

```bash
python -m venv venv
source venv/bin/activate  # Windows: venv\Scripts\activate
```

3. **Install Requirements**

```bash
pip install -r requirements.txt
```

4. **Download YOLOv11 Weights from training notebook**

Put the `best.pt` model inside the `models/` directory.

5. **Run the Pipeline**

```bash
python main.py
```

---

## Results

*  High accuracy in player detection and tracking.
*  Real-time distance and speed metrics rendered on the video.
*  Challenges with detecting the ball due to motion blur and occlusion.
*  Limitations with memory on non-GPU devices.

---


## Limitations

*  Limited real-time capability on non-GPU machines.
*  Ball detection is less reliable.
*  Model performance constrained by dataset quality and size.

## Future Directions

* Improve ball tracking with interpolation techniques.
* Extend to real-time live-stream processing.
* Add tactical heatmap generation and pass-mapping.

---

## Author

**Austin Thapa**

---

## License

This project is licensed under the MIT License - see the [LICENSE](LICENSE) file for details.


