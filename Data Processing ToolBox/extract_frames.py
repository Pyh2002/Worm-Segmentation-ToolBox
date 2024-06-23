import cv2
import os


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

        print(f"Writing frame {frame_count:04d}")
        frame_filename = f"{output_dir}/frame_{frame_count:04d}.png"
        cv2.imwrite(frame_filename, frame)
        frame_count += 1

    cap.release()
