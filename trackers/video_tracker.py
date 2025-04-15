import os
import sys
import pickle
import cv2 as cv
import supervision as sv
import numpy as np

from ultralytics import YOLO

sys.path.append('../')
from utils import get_center_bbox, get_width_bbox, get_foot_position

class Tracker:
    def __init__(self, model_path):
        self.model = YOLO(model_path)
        self.tracker = sv.ByteTrack()
    
    # Make YOLO prediction on each frame to detect players, refrees, goalkeepers    
    def detect_frames(self, frames):
        batch_size = 16
        detections = []      
        for i in range(0, len(frames), batch_size):
            detection_batch = self.model.predict(frames[i:i+batch_size])
            detections += detection_batch
            
        return detections
    
    # Run supervision detection on each frame and track individual objectss
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
            'players': [],
            'referees': [],
            'football': [],
            'goalkeeper': [],
        }
                
        for frame_num, frame in enumerate(detections):
            
            # Detection with supervision
            detection_sv = sv.Detections.from_ultralytics(frame)
            
            # Track objects
            detection_with_tracks = self.tracker.update_with_detections(detection_sv)

            # Ensure that tracks list have at least one empty dictionary to begin with otherwise index error
            tracks["players"].append({})
            tracks["referees"].append({})
            tracks["football"].append({})
            tracks['goalkeeper'].append({})
            
            for frame_detection in detection_with_tracks:          
                bbox = frame_detection[0].tolist()
                track_id = frame_detection[4]
                class_name = frame_detection[5]['class_name']

                if class_name == 'player':
                    tracks['players'][frame_num][track_id] = {'bbox': bbox}
                if class_name == 'referee':
                    tracks['referees'][frame_num][track_id] = {'bbox': bbox}
                if class_name == 'goalkeeper':
                    tracks['goalkeeper'][frame_num][track_id] = {'bbox': bbox}
                if class_name == 'ball':
                    tracks['football'][frame_num][track_id] = {'bbox': bbox}
        try:
            if stub_path is not None:
                with open(stub_path, 'wb') as file:
                    pickle.dump(tracks, file)
                return tracks
        except IOError as e:
            print(f'Error saving the file {stub_path}: ')
    
    def add_position_to_tracks(self, tracks):
        for object, object_tracks in tracks.items():
            for frame_num, track in enumerate(object_tracks):
                for track_id, track_info in track.items():
                    bbox = track_info['bbox']
                    if object == 'football':
                        position = get_center_bbox(bbox)
                    else:
                        position = get_foot_position(bbox)
                    tracks[object][frame_num][track_id]['position']= position
                    
    # Draw the ellipse at the bottom of bounding box with tracker id for each player
    def draw_ellipse(self, frame, bbox, color, track_id):
        
        # Ellipse parameters
        x_center, _ = get_center_bbox(bbox)
        width = get_width_bbox(bbox)
        y2 = bbox[3]
        
        # Draw an ellipse
        cv.ellipse(frame,
                   center=(int(x_center), int(y2)),
                   axes=(int(0.9 * width), int(0.35 * width)),
                   angle=0.0,
                   startAngle=-45,
                   endAngle=245,
                   color=color,
                   lineType=cv.LINE_4)
        
        # Rectangle Parameters
        rect_width = 25
        rect_height = 15
        x1_rect = x_center - rect_width // 2
        x2_rect = x_center + rect_width // 2
        
        y1_rect = (y2 - rect_height //2) + 10
        y2_rect = (y2 + rect_height // 2) + 10
        
        if track_id is not None:
            cv.rectangle(
                frame, 
                (int(x1_rect), int(y1_rect)),
                (int(x2_rect), int(y2_rect)),
                color,
                cv.FILLED
            )

            # Align the text with the rectangle above
            if track_id < 10:
                x1_text = x1_rect + 7
            elif track_id > 99:
                x1_text = x1_rect - 5
            else:
                x1_text = x1_rect + 2
                    
            cv.putText(
                frame,
                f'{track_id}', 
                (int(x1_text), int(y1_rect+11)),
                cv.FONT_HERSHEY_DUPLEX,
                0.5,
                (0,0,0),
                1
            )
        return frame
    
    def draw_triangle(self, frame, bbox, color):
        y = int(bbox[1])
        x, _ = get_center_bbox(bbox)
        
        triangle_points = np.array([
            [[x, y]],
            [[x-10, y-20]],
            [[x+10, y-20]]
        ])
        
        cv.fillPoly(frame, [triangle_points], color)
        cv.polylines(frame, [triangle_points], isClosed=True, color=(0, 0, 0), thickness=2)

    
    def draw_annotations(self, video_frames, tracks):
        output_video_frames = []
        
        # Loop through the frame to draw the annotations around the players
        for frame_num, frame in enumerate(video_frames):
            frame = frame.copy()
            players_dict = tracks['players'][frame_num]
            referees_dict = tracks['referees'][frame_num]
            goalkeeper_dict = tracks['goalkeeper'][frame_num]
            football_dict = tracks['football'][frame_num]
                
            # Draw the players
            for track_id, player in players_dict.items():
                color = player.get('team_color', (0, 0, 255))
                frame = self.draw_ellipse(frame, player['bbox'], color, track_id)
                
            # Draw the refreees
            for track_id, referee in referees_dict.items():
                frame = self.draw_ellipse(frame, referee['bbox'], (0, 255, 255), track_id)
                
            # Draw the goalkeeper
            for track_id, goalkeeper in goalkeeper_dict.items():
                frame = self.draw_ellipse(frame, goalkeeper['bbox'], (255, 255, 0), track_id)
                
            # Draw the football
            for track_id, football in football_dict.items():
                frame = self.draw_triangle(frame, football['bbox'], (0, 255, 0))
            
            # Append the annotated frames into ouput video frames
            output_video_frames.append(frame)
        return output_video_frames