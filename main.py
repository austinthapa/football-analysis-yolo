import cv2 as cv
from utils import read_video, save_video
from trackers import Tracker
from team_assigner import TeamAssigner
from camera_movement_estimator import CameraMovementEstimator
from perspective_transformer import PerspectiveTransformer
from player_metrics import  PlayerMetrics

def main():
    
    # Read video
    video_path = "/Users/anilthapa/football-analysis-yolo/input_videos/long_video.mp4"
    video_frames = read_video(video_path)
    
    # Initialize the Tracker
    model_path = '/Users/anilthapa/football-analysis-yolo/models/best.pt'
    tracker = Tracker(model_path)
    tracks = tracker.get_object_track(video_frames, read_from_stub=True, stub_path='/Users/anilthapa/football-analysis-yolo/stubs/tracks_stub_long.pkl')
    tracker.add_position_to_tracks(tracks)
    
    # Camera movement estimator
    camera_movement_estimator = CameraMovementEstimator(video_frames)
    camera_movement_per_frame = camera_movement_estimator.get_camera_movement(video_frames, read_from_stub=True, stub_path='/Users/anilthapa/football-analysis-yolo/stubs/camera_movement_stub_long.pkl')
    camera_movement_estimator.adjust_positions_to_tracks(tracks, camera_movement_per_frame)
    
    # Perspective Transformer
    perspective_transformer = PerspectiveTransformer()
    perspective_transformer.add_transformed_point_to_tracks(tracks)

    # Calculate player metrics
    player_metrics  = PlayerMetrics()
    player_metrics.add_speed_and_distance_to_tracks(tracks)

    # Assign Players team
    team_assigner = TeamAssigner()
    team_assigner.assign_team_color(video_frames[0], tracks['players'][0])
    
    for frame_num, player_track in enumerate(tracks['players']):
        for player_id, track in player_track.items():
            team = team_assigner.get_player_team(video_frames[frame_num], track['bbox'], player_id)
            tracks['players'][frame_num][player_id]['team'] = team
            tracks['players'][frame_num][player_id]['team_color'] = team_assigner.team_colors[team]
            
    # Draw annotations 
    output_video_frames = tracker.draw_annotations(video_frames, tracks)
    
    # Draw camera movement:
    output_video_frames = camera_movement_estimator.draw_output_frames(output_video_frames, camera_movement_per_frame)
    
    # Draw Speed and Distance
    output_video_frames = player_metrics.draw_speed_and_distance(output_video_frames, tracks)
    
    # Save video
    save_video(output_video_frames, '/Users/anilthapa/football-analysis-yolo/output_videos/output_vid_2.avi')

if __name__ == "__main__":
    main()
