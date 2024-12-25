import os
import sys
import pandas as pd
import numpy as np

from extract_frames import extract_frames
from extract_frames import extract_raw_frames
from create_processed_image import process_image_folder
from create_contour import create_contours_folder
from create_skeleton import create_skeleton_folder
from create_endpoints import create_endpoints_folder
from create_intervals import create_intervals
from create_modified_raw_data import create_modified_raw_data

from clean_file import delete_all_files_in_directory

from calc_speed import calculate_speed
from calc_eccentricity import calculate_eccentricity_angle
from calc_movement import calculate_movement
from calc_movement import generate_movement_graph
from generate_overlays import overlay_folders
from generate_video import images_to_video
from generate_animation import generate_animation
from generate_animation import generate_trace
from calc_video_info import get_video_info


def unitoolMain(parentfolder_path, video, flag='full'):
    video = os.path.normpath(video)
    video_path_name, extension = os.path.splitext(video)
    print(parentfolder_path, video_path_name, extension)

    if (flag == 'partial'):
        contours_dict = create_contours_folder(
            video_path_name + "_processed_frames")
        
        create_modified_raw_data(video_path_name + "_raw_frames",
                                 parentfolder_path + "/raw_data.csv",
                                 parentfolder_path + "/intervals.csv",
                                 contours_dict,
                                 mode='auto')

        overlay_folders(parentfolder_path,
                    video_path_name + "_processed_frames",
                    video_path_name + "_overlayed_images")

        images_to_video(video_path_name + '_overlayed_images',
                    video_path_name + '_output_video.mp4')

    if (flag == 'full'):
        extract_frames(video_path_name, extension)

        process_image_folder(
            video_path_name + "_frames", video_path_name + "_processed_frames")

        contours_dict = create_contours_folder(
            video_path_name + "_processed_frames")

        create_skeleton_folder(video_path_name + "_processed_frames",
                               video_path_name + "_skeletonized_masks")

        create_endpoints_folder(parentfolder_path,
                                video_path_name + "_skeletonized_masks")

        create_intervals(parentfolder_path + "/raw_data.csv",
                         parentfolder_path + "/intervals.csv")

        extract_raw_frames(parentfolder_path + "/intervals.csv",
                           video_path_name + "_raw", extension)

        create_modified_raw_data(video_path_name + "_raw_frames",
                                 parentfolder_path + "/raw_data.csv",
                                 parentfolder_path + "/intervals.csv",
                                 contours_dict,
                                 mode='auto')

        overlay_folders(parentfolder_path,
                    video_path_name + "_processed_frames",
                    video_path_name + "_overlayed_images")

        images_to_video(video_path_name + '_overlayed_images',
                    video_path_name + '_output_video.mp4')

    # Create a processed_data.csv file
    # Copy the first two columns from the raw_data.csv file

    fieldnames = ['frame_number', 'worm_id', 'eccentricity', 'angle']
    raw_data_path = os.path.join(parentfolder_path, "modified_raw_data.csv")
    raw_data = pd.read_csv(raw_data_path)

    interval_data_path = os.path.join(parentfolder_path, "intervals.csv")
    interval_data = pd.read_csv(interval_data_path)

    delete_all_files_in_directory(parentfolder_path)

    for idx, interval in interval_data.iterrows():
        start_frame = interval['start_frame']
        end_frame = interval['end_frame']
        worm_id = interval['worm_id']

        interval_raw_data = raw_data[(raw_data['frame_number'] >= start_frame) & (
            raw_data['frame_number'] <= end_frame)]

        processed_data_path = os.path.join(
            parentfolder_path, f"processed_data_{idx}.csv")
        processed_data = interval_raw_data[interval_raw_data['worm_id'] == worm_id][[
            'frame_number', 'worm_id']].copy()

        for field in fieldnames[2:]:
            processed_data[field] = np.nan

        processed_data = calculate_eccentricity_angle(
            start_frame, end_frame, parentfolder_path, contours_dict, processed_data, worm_id)

        processed_data = calculate_speed(
            start_frame, end_frame, parentfolder_path, processed_data, worm_id)

        processed_data = calculate_movement(
            start_frame, end_frame, parentfolder_path, processed_data, worm_id, threshold=0.1)
        processed_data.to_csv(processed_data_path, index=False)

        generate_movement_graph(idx, start_frame, end_frame,
                                parentfolder_path, video_path_name, worm_id)
        # create_animation(idx, start_frame, end_frame, parentfolder_path, video_path_name)
        generate_trace(idx, start_frame, end_frame,
                       parentfolder_path, video_path_name, worm_id)

    for index in range(len(interval_data)):
        get_video_info(video_path_name, parentfolder_path, index)


if __name__ == "__main__":
    video = sys.argv[1]
    flag = sys.argv[2]
    unitoolMain(os.getcwd(), video, flag)
