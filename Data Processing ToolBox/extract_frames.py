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

        # Convert frame to grayscale
        gray_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        # Apply threshold to get binary mask
        _, binary_mask = cv2.threshold(gray_frame, 127, 255, cv2.THRESH_BINARY)

        print(f"Writing binary mask for frame {frame_count:04d}")
        frame_filename = f"{output_dir}/frame_{frame_count:04d}_mask.png"
        cv2.imwrite(frame_filename, binary_mask)
        frame_count += 1

    cap.release()
