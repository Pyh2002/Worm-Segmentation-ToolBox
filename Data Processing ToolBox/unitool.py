import os
import sys
import pandas as pd
import numpy as np
import cv2
import csv

from extract_frames import extract_frames
from process_image import process_images_and_extract_contours
from extract_skeletons import skeletonize_folder
from generate_endpoints import create_endpoints_folder
from generate_overlays import overlay_folders
from make_video import images_to_video

from calc_speed import calculate_speed
from calc_eccentricity import calculate_eccentricity_angle
from calc_movement import calculate_movement
from calc_movement import create_movement_graph
from create_animation import create_animation
from create_animation import create_trace
from get_video_info import get_video_info


def unitoolMain(subfolder_path, video):
    video = os.path.normpath(video)
    video_name, extension = os.path.splitext(video)
    print(subfolder_path, video_name, extension)

    # extract_frames(video_name, extension)

    # contours = process_images_and_extract_contours(
    #     video_name + "_frames", video_name + "_processed_frames")

    # skeletonize_folder(video_name + "_processed_frames",
    #                    video_name + "_skeletonized_masks")
    # create_endpoints_folder(subfolder_path,
    #                         video_name + "_skeletonized_masks", video_name + "_endpoints")

    # overlay_folders(video_name + "_processed_frames",
    #                 video_name + "_endpoints", video_name + "_overlayed_images")
    # images_to_video(video_name + '_overlayed_images',
    #                 video_name + '_output_video.mp4')

    # Create a processed_data.csv file
    # Copy the first two columns from the raw_data.csv file

    # calculate_eccentricity_angle(subfolder_path, contours)

    # raw_data_path = os.path.join(subfolder_path, "raw_data.csv")
    # processed_data_path = os.path.join(subfolder_path, "processed_data.csv")
    # raw_data = pd.read_csv(raw_data_path)
    # processed_data = pd.read_csv(processed_data_path)
    # processed_data = pd.concat(
    #     [raw_data[['frame_number', 'worm_id']], processed_data], axis=1)
    # processed_data.to_csv(processed_data_path, index=False)

    calculate_speed(subfolder_path)
    calculate_movement(subfolder_path)
    create_animation(subfolder_path, video_name)
    create_movement_graph(subfolder_path, video_name)
    create_trace(subfolder_path, video_name)
    get_video_info(video_name, subfolder_path)


if __name__ == "__main__":
    video = sys.argv[1]
    unitoolMain(os.getcwd(), video)
