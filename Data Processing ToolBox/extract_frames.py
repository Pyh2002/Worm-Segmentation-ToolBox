import cv2
import os
import pandas as pd


def extract_frames(video_name, extension):
    video_path = video_name + extension
    output_dir = video_name + "_frames"
    os.makedirs(output_dir, exist_ok=True)

    cap = cv2.VideoCapture(video_path)

    frame_count = 0
    while True:
        ret, frame = cap.read()
        if not ret:
            break

        gray_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        _, binary_mask = cv2.threshold(gray_frame, 127, 255, cv2.THRESH_BINARY)

        print(f"Writing binary mask for frame {frame_count:04d}")
        frame_filename = f"{output_dir}/frame_{frame_count:04d}_mask.png"
        cv2.imwrite(frame_filename, binary_mask)
        frame_count += 1

    cap.release()


def extract_raw_frames(intervals_path, video_name, extension):
    video_path = video_name + extension
    output_dir = video_name + "_frames"

    intervals = pd.read_csv(intervals_path)
    os.makedirs(output_dir, exist_ok=True)

    cap = cv2.VideoCapture(video_path)

    frame_count = 0
    while True:
        ret, frame = cap.read()
        if not ret:
            break

        if frame_count in intervals['start_frame'].values:
            print(f"Writing raw frame for frame {frame_count:04d}")
            frame_filename = f"{output_dir}/frame_{frame_count:04d}.png"
            cv2.imwrite(frame_filename, frame)
        frame_count += 1

    cap.release()


if __name__ == "__main__":
    import sys
    if len(sys.argv) != 2:
        print("Usage: python extract_frames.py <video_name>")
        sys.exit(1)
    video_name, extension = os.path.splitext(sys.argv[1])
    extract_frames(video_name, extension)
