import os
import sys

from unitool import unitoolMain


def folder_process(folder):
    for video in os.listdir(folder):
        if video.endswith('.avi'):
            video_name, extension = os.path.splitext(video)
            parentfolder_path = os.path.join(folder, video_name)
            os.makedirs(parentfolder_path, exist_ok=True)
            old_video_path = os.path.join(folder, video)
            new_video_path = os.path.join(parentfolder_path, video)
            os.rename(old_video_path, new_video_path)
            print(parentfolder_path, new_video_path)
            unitoolMain(parentfolder_path, new_video_path)


if __name__ == "__main__":
    folder_process(sys.argv[1])
