import cv2
import os
import numpy as np
import pandas as pd


def calculate_eccentricity_angle(start_frame, end_frame, parentfolder_path, contours_dict, processed_data, worm_id):
    raw_data_path = os.path.join(parentfolder_path, "modified_raw_data.csv")
    raw_data = pd.read_csv(raw_data_path)
    
    begin_index = raw_data[raw_data['frame_number'] == start_frame].index[0]
    end_index = raw_data[raw_data['frame_number'] == end_frame].index[-1]

    interval_data = raw_data.loc[begin_index:end_index]
    interval_data = interval_data[interval_data['worm_status'] != 'Coiling or Splitted']
    interval_data = interval_data[interval_data['worm_id'] == worm_id]
    

    def calculate_eccentricity_and_angle(row):
        frame_number = row['frame_number']

        single_contour = contours_dict[frame_number][worm_id - 1]
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

        return pd.Series({'eccentricity': ecc, 'angle': ang})

    # Apply the function to each row in the DataFrame
    processed_data[['eccentricity', 'angle']] = interval_data.apply(calculate_eccentricity_and_angle, axis=1)

    return processed_data
