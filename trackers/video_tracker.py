import os
import pickle
import cv2 as cv
import supervision as sv

from ultralytics import YOLO

class Tracker:
    def __init__(self, model_path):
        self.model = YOLO(model_path)
        self.tracker = sv.ByteTrack()
        
    def detect_frames(self, frames):
        batch_size = 16             
        detections = []      
        for i in range(0, len(frames), batch_size):
            detection = self.model.predict(frames[i:i+batch_size])
            detections.append(detection)
        return detections
    
    def get_object_track(self, frames, read_from_stub = False, stub_path = None):
        
        # Read from pickle if it already exists
        if read_from_stub and stub_path is not None:
            if os.path.exists(stub_path):
                try:
                    with open(stub_path, 'rb') as file:
                        tracks = pickle.load(file)
                    return tracks
                except (FileExistsError, pickle.UnpicklingError) as e:
                    print(f'Error loading the file {stub_path}: {e}')
                    return None

        detections = self.detect_frames(frames)
        tracks = {
            'players': [{}] * len(detections[0]), 
            'refrees': [{}] * len(detections[0]), 
            'football': [{}] * len(detections[0]), 
            'goalkeeper': [{}] * len(detections[0]), 
        }
        # Ensure that tracks list have at least one empty dictionary to begin with otherwise index error
        
        
        detections = detections[0]
        for frame_num, frame in enumerate(detections):

            # Detection with supervision
            detection_sv = sv.Detections.from_ultralytics(frame)
            
            # Track objects
            detection_with_tracks = self.tracker.update_with_detections(detection_sv)
            for frame_detection in detection_with_tracks:
                bbox = frame_detection[0].tolist()
                track_id = frame_detection[4]
                class_name = frame_detection[5]['class_name']
                if class_name == 'player':
                    tracks['players'][frame_num][track_id] = {'bbox': bbox}
                if class_name == 'refree':
                    tracks['refrees'][frame_num][track_id] = {'bbox': bbox}
                if class_name == 'goalkeeper':
                    tracks['goalkeeper'][frame_num][track_id] = {'bbox': bbox}
                if class_name == 'football':
                    tracks['football'][frame_num][track_id] = {'bbox': bbox}
        print(tracks)
        try:
            if stub_path is not None:
                with open(stub_path, 'wb') as file:
                    pickle.dump(tracks, file)
                return tracks
        except IOError as e:
            print(f'Error saving the file {stub_path}: ')