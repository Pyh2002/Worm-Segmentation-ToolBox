import os
import cv2
import numpy as np


def get_contours(binary_mask_image_path):
    image = cv2.imread(binary_mask_image_path, cv2.IMREAD_GRAYSCALE)
    _, binary_mask = cv2.threshold(image, 128, 255, cv2.THRESH_BINARY)
    contour, hierarchy = cv2.findContours(
        binary_mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    contour_image = np.zeros_like(binary_mask)
    cv2.drawContours(contour_image, contour, -1, (255, 255, 255), 1)
    return contour, contour_image


def get_contours_folder(input_folder_path, output_folder_path):
    sorted_file_names = sorted(os.listdir(input_folder_path))
    contours = []
    for file_name in sorted_file_names:
        print(file_name + " contouring")
        if file_name.endswith(".png"):
            if not os.path.exists(output_folder_path):
                os.makedirs(output_folder_path)
            input_image_path = os.path.join(input_folder_path, file_name)
            output_image_path = os.path.join(output_folder_path, file_name)
            contour, contour_image = get_contours(input_image_path)
            contours.append(contour)
            cv2.imwrite(output_image_path, contour_image)
    return contours
