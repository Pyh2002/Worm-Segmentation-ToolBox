import cv2
import os
import numpy as np
import pandas as pd


def get_dilation_AOI(image):
    kernel = np.ones((30, 30), np.uint8)
    dilated_image = cv2.dilate(image, kernel, iterations=1)
    return dilated_image


def dilate_images_AOI_folder(input_folder_path, output_folder_path):
    sorted_file_names = sorted(os.listdir(input_folder_path))
    for file_name in sorted_file_names:
        if file_name.endswith(".png"):
            if not os.path.exists(output_folder_path):
                os.makedirs(output_folder_path)
            input_image_path = os.path.join(input_folder_path, file_name)
            output_image_path = os.path.join(output_folder_path, file_name)
            img = cv2.imread(input_image_path, cv2.IMREAD_GRAYSCALE)
            dilated_image = get_dilation_AOI(img)
            cv2.imwrite(output_image_path, dilated_image)
    return None


def get_rectangle_AOI(image):
    non_zero_pixels = np.argwhere(image > 0)
    if len(non_zero_pixels) == 0:
        return None
    left_most = max(0, np.min(non_zero_pixels[:, 1]) - 30)
    right_most = min(image.shape[1], np.max(non_zero_pixels[:, 1]) + 30)
    top_most = max(0, np.min(non_zero_pixels[:, 0]) - 30)
    bottom_most = min(image.shape[0], np.max(non_zero_pixels[:, 0]) + 30)
    rectangle_AOI = np.zeros_like(image)
    cv2.rectangle(rectangle_AOI, (left_most, top_most),
                  (right_most, bottom_most), 255, -1)
    return rectangle_AOI


def rectangle_AOI_folder(input_folder_path, output_folder_path):
    sorted_file_names = sorted(os.listdir(input_folder_path))
    for file_name in sorted_file_names:
        if file_name.endswith(".png"):
            if not os.path.exists(output_folder_path):
                os.makedirs(output_folder_path)
            input_image_path = os.path.join(input_folder_path, file_name)
            output_image_path = os.path.join(output_folder_path, file_name)
            img = cv2.imread(input_image_path, cv2.IMREAD_GRAYSCALE)
            rectangle_AOI = get_rectangle_AOI(img)
            if rectangle_AOI is None:
                cv2.imwrite(output_image_path, np.zeros_like(img))
                continue
            cv2.imwrite(output_image_path, rectangle_AOI)
    return None


if __name__ == "__main__":
    dilate_images_AOI_folder("WormPartcls-3.5-NaiveTrain1-2A_frames",
                             "WormPartcls-3.5-NaiveTrain1-2A_frames_dilated_AOI")
    rectangle_AOI_folder("WormPartcls-3.5-NaiveTrain1-2A_frames",
                         "WormPartcls-3.5-NaiveTrain1-2A_frames_rectangled_AOI")
