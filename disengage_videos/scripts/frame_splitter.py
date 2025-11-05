import cv2
import os
from pathlib import Path

def extract_stereo_frames(video_path, output_dir, start_time, end_time, target_height, gap):
    """
    Extract frames from a stereo video and split them into left/right images.

    Parameters:
    -----------
    video_path : str
        Path to the MP4 video file
    output_dir : str
        Directory to save extracted frames
    start_time : float
        Start time in seconds
    end_time : float
        End time in seconds
    target_height : int
        Desired height of the output images
    gap : int
        Number of frames to skip between extractions (default=1 means every frame)
    """

    output_path = Path(output_dir)
    output_path.mkdir(parents=True, exist_ok=True)

    cap = cv2.VideoCapture(video_path)
    if not cap.isOpened():
        print(f"Error: Cannot open video file {video_path}")
        return

    fps = cap.get(cv2.CAP_PROP_FPS)
    total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    duration = total_frames / fps
    width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))

    print(f"Video properties:")
    print(f"  Resolution: {width}x{height}")
    print(f"  FPS: {fps:.2f}")
    print(f"  Duration: {duration:.2f} seconds")
    print(f"  Total frames: {total_frames}")

    if start_time < 0 or end_time > duration or start_time >= end_time:
        print(f"Error: Invalid time range. Video duration is {duration:.2f} seconds")
        return

    start_frame = int(start_time * fps)
    end_frame = int(end_time * fps)

    print(f"\nExtracting frames {start_frame} to {end_frame} (every {gap} frame(s))")  ### CHANGED
    print(f"Time range: {start_time:.2f}s to {end_time:.2f}s")

    cap.set(cv2.CAP_PROP_POS_FRAMES, start_frame)

    frame_count = 0
    saved_count = 0  ### CHANGED

    while True:
        ret, frame = cap.read()
        if not ret or (start_frame + frame_count) >= end_frame:
            break

        # --- Only process every Nth frame ---
        if frame_count % gap == 0:  ### CHANGED
            mid_x = width // 2
            left_frame = frame[:, :mid_x]
            right_frame = frame[:, mid_x:]

            aspect_ratio = width / height
            target_width = int(target_height * aspect_ratio)
            left_frame = cv2.resize(left_frame, (target_width, target_height))
            right_frame = cv2.resize(right_frame, (target_width, target_height))

            left_path = output_path / f"frame_{saved_count:06d}_left.png"
            right_path = output_path / f"frame_{saved_count:06d}_right.png"

            cv2.imwrite(str(left_path), left_frame)
            cv2.imwrite(str(right_path), right_frame)
            saved_count += 1  ### CHANGED

            if saved_count % 10 == 0:
                print(f"  Saved {saved_count} frame pairs...", end='\r')

        frame_count += 1  ### CHANGED

    cap.release()
    print(f"\n\nComplete! Extracted {saved_count} frame pairs")
    print(f"Frames saved to: {output_path}")

if __name__ == "__main__":
    # Configuration
    video_path = "/home/jonathan/Github/FSR_DLNR/disengage_videos/20160211_163956_603_001.mp4"  # Change this to your video path
    output_dir = "/home/jonathan/Github/FSR_DLNR/disengage_videos/frames"        # Output directory
    start_time = 15.0                         
    end_time = 60+15                          
    
    # Extract frames
    extract_stereo_frames(video_path, output_dir, start_time, end_time, target_height = 640, gap = 10)
