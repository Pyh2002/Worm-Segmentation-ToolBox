import os
import sys

from unitool import unitoolMain


def folder_process(folder):
    for video in os.listdir(folder):
        if video.endswith('.avi'):
            video_name, extension = os.path.splitext(video)
            subfolder_path = os.path.join(folder, video_name)
            os.makedirs(subfolder_path, exist_ok=True)
            old_video_path = os.path.join(folder, video)
            new_video_path = os.path.join(subfolder_path, video)
            os.rename(old_video_path, new_video_path)
            print(subfolder_path, new_video_path)
            unitoolMain(subfolder_path, new_video_path)


if __name__ == "__main__":
    folder_process(sys.argv[1])
