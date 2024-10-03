import cv2
import os
import numpy as np
import pandas as pd


def calculate_eccentricity_angle(start_frame, end_frame, parentfolder_path, contours_dict, processed_data):
    raw_data_path = os.path.join(parentfolder_path, "modified_raw_data.csv")
    raw_data = pd.read_csv(raw_data_path)
    start_index = raw_data[raw_data['frame_number'] == start_frame].index[0]
    end_index = raw_data[raw_data['frame_number'] == end_frame].index[-1]
    
    # print("len(contours_dict): ", len(contours_dict))
    # print("start_index: ", start_index)
    # print("end_index: ", end_index)
    
    for index in range(start_index, end_index + 1):
        frame_number = raw_data.loc[index, 'frame_number']
        worm_id = raw_data.loc[index, 'worm_id']
        for j in range(len(contours_dict[frame_number])):
            if (worm_id == 0):
                processed_data.loc[(processed_data['frame_number'] == frame_number) &
                                   (processed_data['worm_id'] == worm_id), 'eccentricity'] = 0
                processed_data.loc[(processed_data['frame_number'] == frame_number) &
                                   (processed_data['worm_id'] == worm_id), 'angle'] = 0
                continue

            single_contour = contours_dict[frame_number][j]
            moments = cv2.moments(single_contour)
            mu11 = moments["mu11"]
            mu02 = moments["mu02"]
            mu20 = moments["mu20"]

            a1 = (mu20 + mu02) / 2
            a2 = np.sqrt(4 * mu11**2 + (mu20 - mu02)**2) / 2

            minor_axis = a1 - a2
            major_axis = a1 + a2

            if major_axis == 0:
                ecc = 0
            else:
                ecc = np.sqrt(1 - minor_axis / major_axis)
            ang = np.arctan2(2 * mu11, (mu20 - mu02)) / 2
            ang *= 180 / np.pi

            processed_data.loc[(processed_data['frame_number'] == frame_number) &
                               (processed_data['worm_id'] == worm_id), 'eccentricity'] = ecc
            processed_data.loc[(processed_data['frame_number'] == frame_number) &
                               (processed_data['worm_id'] == worm_id), 'angle'] = ang

    return processed_data
