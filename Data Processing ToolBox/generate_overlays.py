import numpy as np
from PIL import Image
import os


def overlay_images(base_image_path, overlay_image_path, output_image_path):
    base_image = Image.open(base_image_path).convert("RGBA")
    overlay_image = Image.open(overlay_image_path).convert("RGBA")
    if base_image.size != overlay_image.size:
        overlay_image = overlay_image.resize(base_image.size, Image.ANTIALIAS)

    r, g, b, a = overlay_image.split()
    r = np.array(r)
    g = np.array(g)
    b = np.array(b)

    mask = np.where((r != 0) | (g != 0) | (b != 0), 255, 0).astype(np.uint8)
    overlay_image.putalpha(Image.fromarray(mask))
    base_image.paste(overlay_image, (0, 0), overlay_image)
    base_image.save(output_image_path)


def overlay_folders(input_folder_path, overlay_folder_path, output_folder_path):
    sorted_base_file_names = sorted(os.listdir(input_folder_path))
    sorted_overlay_file_names = sorted(os.listdir(overlay_folder_path))

    for base_file_name, overlay_file_name in zip(sorted_base_file_names, sorted_overlay_file_names):
        if base_file_name.endswith(".png") and overlay_file_name.endswith(".png"):
            print(base_file_name + " overlaying")
            if not os.path.exists(output_folder_path):
                os.makedirs(output_folder_path)
            base_image_path = os.path.join(input_folder_path, base_file_name)
            overlay_image_path = os.path.join(
                overlay_folder_path, overlay_file_name)
            output_image_path = os.path.join(
                output_folder_path, base_file_name)
            overlay_images(base_image_path, overlay_image_path,
                           output_image_path)
