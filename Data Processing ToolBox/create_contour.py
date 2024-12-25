import cv2
import os
import numpy as np


def create_contours(image):
    contours, _ = cv2.findContours(
        image, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    return contours

def create_contours_folder(input_folder_path):
    sorted_file_names = sorted(os.listdir(input_folder_path))
    contours_dict = {}
    for index, file_name in enumerate(sorted_file_names):
        if file_name.endswith(".png"):
            print(file_name + " creating contours")
            input_image_path = os.path.join(input_folder_path, file_name)
            img = cv2.imread(input_image_path, cv2.IMREAD_GRAYSCALE)
            contours = create_contours(img)
            contours_dict[index] = contours
    return contours_dict

if __name__ == "__main__":
    import sys
    if len(sys.argv) != 3:
        print("Usage: python create_contours.py <input_folder_path> <output_folder_path>")
        sys.exit(1)
    create_contours_folder(sys.argv[1], sys.argv[2])
