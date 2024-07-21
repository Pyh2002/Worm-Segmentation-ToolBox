import cv2
import os
import numpy as np


def extract_contours(image):
    contours, _ = cv2.findContours(
        image, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    return contours


def process_erosion_dilation(image):
    contours = extract_contours(image)
    max_area = 0
    for contour in contours:
        area = cv2.contourArea(contour)
        print(area)
        if area > max_area:
            max_area = area

    for (i, contour) in enumerate(contours):
        if cv2.contourArea(contour) < max_area / 2:
            cv2.drawContours(image, contours, i, 0, -1)

    kernel = np.ones((10, 10), np.uint8)
    dilated_image = cv2.dilate(image, kernel, iterations=1)
    eroded_image = cv2.erode(dilated_image, kernel, iterations=1)
    return eroded_image


def process_images_and_extract_contours(input_folder_path, output_folder_path):
    sorted_file_names = sorted(os.listdir(input_folder_path))
    contours_list = []
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
            contours = extract_contours(processed_image)
            contours_list.append(contours)
    return contours_list
