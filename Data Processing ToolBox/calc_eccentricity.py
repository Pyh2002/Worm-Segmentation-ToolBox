import cv2
import os
import numpy as np
import csv


def calculate_eccentricity_angle(subfolder_path, contours):
    data = []

    for contour in contours:
        # if len(contour) >= 2:
        #     data.append({'eccentricity': None, 'angle': None})
        # else:
        for single_contour in contour:
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

            data.append({'eccentricity': ecc, 'angle': ang})

    fieldnames = ['eccentricity', 'angle']

    processed_data_path = os.path.join(subfolder_path, 'processed_data.csv')
    with open(processed_data_path, 'w', newline='') as file:
        writer = csv.DictWriter(file, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(data)
