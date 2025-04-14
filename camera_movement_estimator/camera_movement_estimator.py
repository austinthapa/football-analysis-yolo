import os
import sys
import pickle
import numpy as np
import cv2 as cv

sys.path.append('../')
from utils import measure_distance, measure_x_y_distance

class CameraMovementEstimator:
    
    def __init__(self, frames):
        first_frame_gray = cv.cvtColor(frames[0], cv.COLOR_BGR2GRAY)
        mask_features = np.zeros_like(first_frame_gray)
        mask_features[:50, :] = 1
        mask_features[-50:, :] = 1
        self.minimum_camera_dist = 5
        self.features_to_track = dict(
            maxCorners = 100, 
            qualityLevel = 0.3,
            minDistance = 3, 
            blockSize = 7, 
            mask = mask_features
        )
        self.lk_params = dict(
            winSize = (15, 15), 
            maxLevel = 2, 
            criteria = (cv.TERM_CRITERIA_EPS | cv.TERM_CRITERIA_COUNT, 10, 0.3)
        )
        
        
    def get_camera_movement(self, frames, read_from_stub = False, stub_path = None):
        
        # Read from the stub
        if read_from_stub and stub_path is not None:
            if os.path.exists(stub_path):
                try:
                    with open(stub_path, 'rb') as file:
                        camera_movement = pickle.load(file)
                    return camera_movement
                except (FileExistsError, pickle.UnpicklingError) as e:
                    print(f'Error loading file at: {stub_path}: {e}')
                    return None
                
        camera_movement = [[0, 0] for _ in range(len(frames))]
        old_frame_gray = cv.cvtColor(frames[0], cv.COLOR_BGR2GRAY)
        old_features = cv.goodFeaturesToTrack(old_frame_gray, **self.features_to_track)
        
        # Loop through the frames skipping the first one
        for frame_num in range(1, len(frames)):
            new_frame_gray = cv.cvtColor(frames[frame_num], cv.COLOR_BGR2GRAY)
            new_features, status, _ = cv.calcOpticalFlowPyrLK(old_frame_gray, new_frame_gray, old_features, None, **self.lk_params)
            max_distance = 0
            
            camera_movement_x, camera_movement_y = 0, 0
            for (old_feature, new_feature) in zip(old_features, new_features):
                old_feature_point = old_feature.ravel()
                new_feature_point = new_feature.ravel()
                distance = measure_distance(old_feature_point, new_feature_point)
                if distance > max_distance:
                    max_distance = distance
                    camera_movement_x, camera_movement_y = measure_x_y_distance(old_feature_point, new_feature_point)
            
            if max_distance > self.minimum_camera_dist:
                camera_movement[frame_num] = [camera_movement_x, camera_movement_y]
                old_features = cv.goodFeaturesToTrack(new_frame_gray, **self.features_to_track)
            old_frame_gray = new_frame_gray.copy()
        
        # Save into new stubs
        try:
            with open(stub_path, 'wb') as file:
                pickle.dump(camera_movement, file)
            return camera_movement
        except IOError as e:
            print(f'Error occured:{e}')
            
        
    # Draw the camera movement in the output frames
    def draw_output_frames(self, frames, camera_movement):
        output_frames = []
        for frame_num, frame in enumerate(frames):
            frame = frame.copy()
            overlay = frame.copy()
            
            # Draw a  semi-rectange on the top left
            cv.rectangle(overlay, (0, 0), (500, 100), (255, 255, 255), -1)
            alpha = 0.6
            frame = cv.addWeighted(overlay, alpha, frame, 1-alpha, 0)
            
            # Get the camera X movement and Y movement
            x_movement, y_movement = camera_movement[frame_num]
            
            # Put the text inside the rectangle
            cv.putText(frame, f'Camera X movement: {x_movement:.2f}',(10, 30), cv.FONT_HERSHEY_PLAIN, 1.5, (0, 0, 0), 2)
            cv.putText(frame, f'Camera Y movement: {y_movement:.2f}', (20, 30), cv.FONT_HERSHEY_PLAIN, 1.5, (0, 0, 0), 2)
            
            output_frames.append(frame)            
            
        return output_frames