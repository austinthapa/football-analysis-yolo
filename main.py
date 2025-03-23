from utils import read_video, save_video

def main():
    
    # Read video
    video_path = "/Users/anilthapa/football-analysis-yolo/input_videos/input_video.mp4"
    video_frames = read_video(video_path)
    
    # Run the tracker
    
    
    # Save video
    save_video(video_frames, '/Users/anilthapa/football-analysis-yolo/output_videos/output.avi')

if __name__ == "__main__":
    main()
