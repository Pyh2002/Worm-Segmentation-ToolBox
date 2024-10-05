import os
import sys
import pandas as pd
import numpy as np
import cv2
import csv

from extract_frames import extract_frames
from extract_frames import extract_raw_frames
from process_image import process_images_and_extract_contours
from extract_skeletons import skeletonize_folder
from generate_endpoints import create_endpoints_folder
from group_images import group_images
from process_intervals import process_intervals
from generate_overlays import overlay_folders
from make_video import images_to_video

from calc_speed import calculate_speed
from calc_eccentricity import calculate_eccentricity_angle
from calc_movement import calculate_movement
from calc_movement import create_movement_graph
from create_animation import create_animation
from create_animation import create_trace
from get_video_info import get_video_info


def unitoolMain(parentfolder_path, video):
    video = os.path.normpath(video)
    video_path_name, extension = os.path.splitext(video)
    print(parentfolder_path, video_path_name, extension)

    # extract_frames(video_path_name, extension)
    # extract_raw_frames(video_path_name + "_raw", extension)

    # contours_dict = process_images_and_extract_contours(
    #     video_path_name + "_frames", video_path_name + "_processed_frames")

    # skeletonize_folder(video_path_name + "_processed_frames",
    #                    video_path_name + "_skeletonized_masks")

    # create_endpoints_folder(parentfolder_path,
    #                         video_path_name + "_skeletonized_masks")

    # group_images(parentfolder_path + "/raw_data.csv",
    #              parentfolder_path + "/intervals.csv")

    process_intervals(video_path_name + "_raw_frames",
                      parentfolder_path + "/raw_data.csv",
                      parentfolder_path + "/intervals.csv",
                      mode='manual')

    overlay_folders(parentfolder_path,
                    video_path_name + "_processed_frames",
                    video_path_name + "_overlayed_images")

    images_to_video(video_path_name + '_overlayed_images',
                    video_path_name + '_output_video.mp4')

    # Create a processed_data.csv file
    # Copy the first two columns from the raw_data.csv file

    fieldnames = ['frame_number', 'worm_id', 'eccentricity', 'angle']
    raw_data_path = os.path.join(parentfolder_path, "raw_data.csv")
    raw_data = pd.read_csv(raw_data_path)

    interval_data_path = os.path.join(parentfolder_path, "intervals.csv")
    interval_data = pd.read_csv(interval_data_path)

    for idx, interval in interval_data.iterrows():
        start_frame = interval['start_frame']
        end_frame = interval['end_frame']

        interval_data = raw_data[(raw_data['frame_number'] >= start_frame) & (
            raw_data['frame_number'] <= end_frame)]

        processed_data_path = os.path.join(
            parentfolder_path, f"processed_data_{idx}.csv")
        processed_data = interval_data[['frame_number', 'worm_id']].copy()

        for field in fieldnames[2:]:
            processed_data[field] = np.nan

        processed_data = calculate_eccentricity_angle(
            start_frame, end_frame, parentfolder_path, contours_dict, processed_data)
        processed_data = calculate_speed(
            start_frame, end_frame, parentfolder_path, processed_data)
        processed_data = calculate_movement(
            start_frame, end_frame, parentfolder_path, processed_data)
        processed_data.to_csv(processed_data_path, index=False)
        create_movement_graph(idx, start_frame, end_frame,
                              parentfolder_path, video_path_name)
        # create_animation(idx, start_frame, end_frame, parentfolder_path, video_path_name)
        create_trace(idx, start_frame, end_frame,
                     parentfolder_path, video_path_name)

    # get_video_info(video_path_name, parentfolder_path)


if __name__ == "__main__":
    video = sys.argv[1]
    unitoolMain(os.getcwd(), video)
