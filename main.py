from utils import read_video, save_video
from trackers import Tracker

def main():
    
    # Read video
    video_path = "/Users/anilthapa/football-analysis-yolo/input_videos/demo.mp4"
    video_frames = read_video(video_path)
    
    # Initialize the Tracker
    model_path = '/Users/anilthapa/football-analysis-yolo/models/best.pt'
    tracker = Tracker(model_path)
    tracks = tracker.get_object_track(video_frames, read_from_stub=True, stub_path='/Users/anilthapa/football-analysis-yolo/stubs/tracks_stub.pkl')
    print(tracks['players'][5][1])
    # Save video
    save_video(video_frames, '/Users/anilthapa/football-analysis-yolo/output_videos/output.avi')

if __name__ == "__main__":
    main()
