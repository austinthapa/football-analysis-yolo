import os
import numpy as np
import cv2 as cv

class PerspectiveTransformer:
    def __init__(self):
        width = 68
        length = 23.32
        self.pixel_vertices = np.array([
            [110, 1035], 
            [265, 275], 
            [910, 260], 
            [1640, 915]]).astype(np.float32())
        self.target_vertices = np.array([
            [0, width],
            [0, 0],
            [length, 0],
            [length, width]]).astype(np.float32)
        self.perspective_transformer = cv.getPerspectiveTransform(self.pixel_vertices, self.target_vertices)
        
    def transform_point(self, point):
        p = (int(point[0]), int(point[1]))
        is_inside = cv.pointPolygonTest(self.pixel_vertices, p, False) >= 0
        if not is_inside:
            return None
        reshaped_point = point.reshape(-1, 1, 2).astype(np.float32)
        transformed_point = cv.perspectiveTransform(reshaped_point, self.perspective_transformer)
        return transformed_point.reshape(-1, 2)
    
    def add_transformed_point_to_tracks(self, tracks):
        for object, object_tracks in tracks.items():
            for frame_num, track in enumerate(object_tracks):
                for track_id, track_info in track.items():
                    position = track_info['position_adjusted']
                    position = np.array(position)
                    position_transformed = self.transform_point(position)
                    if position_transformed is not None:
                        position_transformed = position_transformed.squeeze().tolist()
                    track[track_id]['position_transformed'] = position_transformed