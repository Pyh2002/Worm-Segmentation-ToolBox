import cv2
import os


def images_to_video(input_folder_path, output_video_path, fps=14.225):
    images = [img for img in sorted(os.listdir(
        input_folder_path)) if img.endswith((".png", ".jpg", ".jpeg"))]
    sample_img_path = os.path.join(input_folder_path, images[0])
    frame = cv2.imread(sample_img_path)
    height, width, layers = frame.shape

    fourcc = cv2.VideoWriter_fourcc(*'mp4v')
    video = cv2.VideoWriter(output_video_path, fourcc, fps, (width, height))
    for image in images:
        img_path = os.path.join(input_folder_path, image)
        frame = cv2.imread(img_path)
        video.write(frame)
    video.release()
