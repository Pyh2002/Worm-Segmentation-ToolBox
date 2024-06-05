import cv2
import os

video_name = "WormParticles45um-3.0-Wrm2"
video_path = video_name + ".avi"
output_dir = video_name + "_frames"
os.makedirs(output_dir, exist_ok=True)

# Open the video
cap = cv2.VideoCapture(video_path)

frame_count = 0
while True:
    ret, frame = cap.read()
    if not ret:
        break

    print(f"Writing frame {frame_count:04d}")
    frame_filename = f"{output_dir}/frame_{frame_count:04d}.png"
    frame = cv2.medianBlur(frame, 5)
    frame = cv2.morphologyEx(frame, cv2.MORPH_OPEN, (5, 5))
    cv2.imwrite(frame_filename, frame)
    frame_count += 1

cap.release()
