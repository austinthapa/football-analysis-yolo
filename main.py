import cv2 as cv
from utils import read_video, save_video
from trackers import Tracker
from team_assigner import TeamAssigner
from camera_movement_estimator import CameraMovementEstimator
from perspective_transformer import PerspectiveTransformer

def main():
    
    # Read video
    video_path = "/Users/anilthapa/football-analysis-yolo/input_videos/input_video.mp4"
    video_frames = read_video(video_path)
    
    # Initialize the Tracker
    model_path = '/Users/anilthapa/football-analysis-yolo/models/best.pt'
    tracker = Tracker(model_path)
    tracks = tracker.get_object_track(video_frames, read_from_stub=False, stub_path='/Users/anilthapa/football-analysis-yolo/stubs/tracks_stub_long.pkl')
    
    # Camera movement estimator
    camera_movement_estimator = CameraMovementEstimator(video_frames)
    camera_movement_per_frame = camera_movement_estimator.get_camera_movement(video_frames, read_from_stub=True, stub_path='/Users/anilthapa/football-analysis-yolo/stubs/camera_movement_stub_short.pkl')
    
    # Perspective Transformer
    perspective_transformer = PerspectiveTransformer()
    perspective_transformer.add_transformed_point_to_tracks(tracks)

    # Assign Players team
    team_assigner = TeamAssigner()
    team_assigner.assign_team_color(video_frames[0], tracks['player'][0])
    
    for frame_num, player_track in enumerate(tracks['player']):
        for player_id, track in player_track.items():
            team = team_assigner.get_player_team(video_frames[frame_num], track['bbox'], player_id)
            tracks['players'][frame_num][player_id]['team'] = team
            tracks['players'][frame_num][player_id]['team_color'] = team_assigner.team_colors[team]
            
    # Draw annotations 
    output_video_frames = tracker.draw_annotations(video_frames, tracks)
    
    # Draw camera movement:
    output_video_frames = camera_movement_estimator.draw_camera_movement(output_video_frames, camera_movement_per_frame)
    
    # Save video
    save_video(output_video_frames, '/Users/anilthapa/football-analysis-yolo/output_videos/output.avi')

if __name__ == "__main__":
    main()
