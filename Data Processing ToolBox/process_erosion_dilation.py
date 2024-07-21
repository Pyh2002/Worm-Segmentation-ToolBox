import cv2
import os
import numpy as np


def process_erosion_dilation(image):
    kernel = np.ones((10, 10), np.uint8)
    dilated_image = cv2.dilate(image, kernel, iterations=1)
    eroded_image = cv2.erode(dilated_image, kernel, iterations=1)
    return eroded_image


def process_erosion_dilation_folder(input_folder_path, output_folder_path):
    sorted_file_names = sorted(os.listdir(input_folder_path))
    for file_name in sorted_file_names:
        print(file_name + " processing")
        if file_name.endswith(".png"):
            if not os.path.exists(output_folder_path):
                os.makedirs(output_folder_path)
            input_image_path = os.path.join(input_folder_path, file_name)
            output_image_path = os.path.join(output_folder_path, file_name)
            img = cv2.imread(input_image_path, cv2.IMREAD_GRAYSCALE)
            processed_image = process_erosion_dilation(img)
            cv2.imwrite(output_image_path, processed_image)
