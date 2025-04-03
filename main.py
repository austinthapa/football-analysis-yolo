import cv2 as cv
from utils import read_video, save_video
from trackers import Tracker
from team_assigner import TeamAssigner

def main():
    
    # Read video
    video_path = "/Users/anilthapa/football-analysis-yolo/input_videos/input_video.mp4"
    video_frames = read_video(video_path)
    
    # Initialize the Tracker
    model_path = '/Users/anilthapa/football-analysis-yolo/models/best.pt'
    tracker = Tracker(model_path)
    tracks = tracker.get_object_track(video_frames, read_from_stub=True, stub_path='/Users/anilthapa/football-analysis-yolo/stubs/tracks_stub_long.pkl')
    
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
    
    # Save video
    save_video(output_video_frames, '/Users/anilthapa/football-analysis-yolo/output_videos/output.avi')

if __name__ == "__main__":
    main()
