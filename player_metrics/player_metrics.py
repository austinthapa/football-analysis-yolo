import sys
import numpy as np
import cv2 as cv

sys.path.append('../')
from utils import measure_distance, get_foot_position

class PlayerMetrics:
    def __init__(self):
        self.frame_rate = 24
        self.frame_window = 5

    def add_speed_and_distance_to_tracks(self, tracks):
        total_distance = {}

        excluded_objects = {'football', 'refrees', 'goalkeeper'}

        for object_type, object_track in tracks.items():
            if object_type in excluded_objects:
                continue

            num_frames = len(object_track)
            total_distance.setdefault(object_type, {})

            for frame_start in range(0, num_frames, self.frame_window):
                frame_end = min(frame_start + self.frame_window, num_frames - 1)

                frame_data_start = object_track[frame_start]
                frame_data_end = object_track[frame_end]

                for track_id in frame_data_start:
                    if track_id not in frame_data_end:
                        continue

                    start_pos = frame_data_start[track_id].get('position_transformed')
                    end_pos = frame_data_end[track_id].get('position_transformed')

                    if not start_pos or not end_pos:
                        continue

                    distance = measure_distance(start_pos, end_pos)
                    time_elapsed = (frame_end - frame_start) / self.frame_rate
                    speed = distance / time_elapsed if time_elapsed > 0 else 0

                    total_distance[object_type][track_id] = total_distance[object_type].get(track_id, 0) + distance

                    for f in range(frame_start, frame_end):
                        if track_id in object_track[f]:
                            object_track[f][track_id]['speed'] = speed
                            object_track[f][track_id]['distance'] = total_distance[object_type][track_id]

    def draw_speed_and_distance(self, frames, tracks):
        excluded_objects = {'football', 'goalkeeper', 'refrees'}
        output_frames = []
        for frame_num, frame in enumerate(frames):
            for object_type, object_track in tracks.items():
                if object_type in excluded_objects:
                    continue

                frame_objects = object_track[frame_num]
                for track_info in frame_objects.values():
                    speed = track_info.get('speed')
                    distance = track_info.get('distance')
                    if speed is None or distance is None:
                        continue

                    bbox = track_info['bbox']
                    position = list(get_foot_position(bbox))
                    position[1] += 40
                    position = tuple(map(int, position))

                    cv.putText(frame, f'Speed: {speed:.2f} m/s', position, cv.FONT_HERSHEY_SIMPLEX, 0.3, (255, 255, 255), 2)
                    position_distance = (position[0], position[1] + 20)
                    cv.putText(frame, f'Distance: {distance:.2f} meters', position_distance, cv.FONT_HERSHEY_SIMPLEX, 0.3, (255, 255, 255), 2)
            output_frames.append(frame)
        return output_frames