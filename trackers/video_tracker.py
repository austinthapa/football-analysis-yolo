import cv2 as cv
import supervision as sv

from ultralytics import YOLO

class Tracker:
    def __init__(self, model_path):
        self.model = YOLO(model_path)
        self.tracker = sv.ByteTrack()
        
    def detect_frames(self, frames):
        batch_size = 3             # Change this into original batch size later on
        detections = []      
        for i in range(0, len(frames), batch_size):
            detection = self.model.predict(frames[i:i+batch_size])
            detections.append(detection)
            break             # Remove this later
        return detections
    
    def get_object_track(self, frames):
        detection = self.detect_frames(frames)
        print(len(detection))