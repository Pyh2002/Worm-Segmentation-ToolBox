from skimage.morphology import skeletonize
import numpy as np
from PIL import Image
import os


def skeletonize_image(image):
    skeleton = skeletonize(image)
    return skeleton


def create_skeleton_folder(input_folder_path, output_folder_path):
    sorted_file_names = sorted(os.listdir(input_folder_path))
    for file_name in sorted_file_names:
        print(file_name + " creating skeleton")
        if file_name.endswith(".png"):
            if not os.path.exists(output_folder_path):
                os.makedirs(output_folder_path)
            input_image_path = os.path.join(input_folder_path, file_name)
            output_image_path = os.path.join(output_folder_path, file_name)
            img = Image.open(input_image_path)
            img_array = np.array(img)

            mask = (img_array > 128).astype(np.uint8)
            skeleton = skeletonize_image(mask)
            skeleton = skeleton.astype(int)
            skeleton *= 255
            skeleton_img = Image.fromarray(skeleton.astype(np.uint8))
            skeleton_img.save(output_image_path)


if __name__ == "__main__":
    import sys
    if len(sys.argv) != 3:
        print("Usage: python create_skeleton.py <input_folder_path> <output_folder_path>")
        sys.exit(1)
    create_skeleton_folder(sys.argv[1], sys.argv[2])
