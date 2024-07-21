import numpy as np
from scipy.signal import convolve2d
from PIL import Image, ImageDraw
from skimage.graph import route_through_array
from skimage.measure import label, regionprops
import csv
import os


def analyze_skeleton(binary_mask):
    try:
        kernel = np.array([[1, 1, 1], [1, 0, 1], [1, 1, 1]])
        neighbor_count = convolve2d(
            binary_mask, kernel, mode="same", boundary="fill", fillvalue=0)
        endpoints = (neighbor_count == 1) & binary_mask
        endpoint_coords = np.argwhere(endpoints)
        if len(endpoint_coords) > 1:
            path, _ = route_through_array(
                1 - binary_mask,
                start=endpoint_coords[0],
                end=endpoint_coords[-1],
                fully_connected=True,
            )
            path = np.array(path)
        else:
            path = np.array([])

        path_length = len(path)
        quartile_indices = np.linspace(0, len(path) - 1, 5, dtype=int)
        quartile_coords = [path[index]
                           for index in quartile_indices if index < len(path)]
        return endpoint_coords, quartile_coords, path_length, path
    except:
        return [], [], 0, []


def process_endpoint_image(image_path):
    img = Image.open(image_path).convert("L")
    binary_mask = np.array(img) > 128
    labeled_mask, num_labels = label(binary_mask, return_num=True)
    img = img.convert("RGB")
    draw = ImageDraw.Draw(img)

    worm_data = []

    if num_labels >= 2:
        worm_data.append({
            "endpoints": None,
            "quartile_coords": None,
            "path_length": None,
            "worm_id": -1
        })
        return worm_data, img

    for region in regionprops(labeled_mask):
        worm_mask = labeled_mask == region.label
        endpoint_coords, quartile_coords, path_length, path = analyze_skeleton(
            worm_mask)

        if len(path) == 0:
            worm_data.append({
                "endpoints": None,
                "quartile_coords": None,
                "path_length": 0,
                "worm_id": 0
            })
            continue

        for y, x in endpoint_coords:
            draw.ellipse((x - 2, y - 2, x + 2, y + 2),
                         outline="red", width=1)

        for y, x in quartile_coords:
            draw.ellipse((x - 2, y - 2, x + 2, y + 2),
                         fill="blue", outline="blue", width=1)

        worm_data.append({
            "endpoints": endpoint_coords,
            "quartile_coords": quartile_coords,
            "path_length": path_length,
            "worm_id": 1
        })

    return worm_data, img


def create_endpoints_folder(subfolder_path, input_folder_path, output_folder_path):
    sorted_file_names = sorted(os.listdir(input_folder_path))
    raw_data_path = os.path.join(subfolder_path, "raw_data.csv")

    fieldnames = ['frame_number', 'worm_id', 'y_head', 'x_head', 'y_neck',
                  'x_neck', 'y_mid', 'x_mid', 'y_hip', 'x_hip', 'y_tail', 'x_tail', 'path_length']
    with open(raw_data_path, 'w', newline='') as file:
        writer = csv.DictWriter(file, fieldnames=fieldnames)
        writer.writeheader()

        for file_name in sorted_file_names:
            print(file_name + " creating endpoints")
            if file_name.endswith(".png"):
                if not os.path.exists(output_folder_path):
                    os.makedirs(output_folder_path)
                input_image_path = os.path.join(input_folder_path, file_name)
                output_image_path = os.path.join(output_folder_path, file_name)
                worm_data, analyzed_img = process_endpoint_image(
                    input_image_path)
                analyzed_img.save(output_image_path)
                frame_number = int(file_name.split("_")[1].split(".")[0])

                for worm in worm_data:
                    if worm["worm_id"] == -1:
                        data = {
                            "frame_number": frame_number,
                            "worm_id": -1,
                            "y_head": None,
                            "x_head": None,
                            "y_neck": None,
                            "x_neck": None,
                            "y_mid": None,
                            "x_mid": None,
                            "y_hip": None,
                            "x_hip": None,
                            "y_tail": None,
                            "x_tail": None,
                            "path_length": None
                        }
                    elif worm["worm_id"] == 0:
                        data = {
                            "frame_number": frame_number,
                            "worm_id": 0,
                            "y_head": None,
                            "x_head": None,
                            "y_neck": None,
                            "x_neck": None,
                            "y_mid": None,
                            "x_mid": None,
                            "y_hip": None,
                            "x_hip": None,
                            "y_tail": None,
                            "x_tail": None,
                            "path_length": 0
                        }
                    else:
                        data = {
                            "frame_number": frame_number,
                            "worm_id": worm["worm_id"],
                            "y_head": int(worm["quartile_coords"][0][0]) if len(worm["quartile_coords"]) > 0 else None,
                            "x_head": int(worm["quartile_coords"][0][1]) if len(worm["quartile_coords"]) > 0 else None,
                            "y_neck": int(worm["quartile_coords"][1][0]) if len(worm["quartile_coords"]) > 1 else None,
                            "x_neck": int(worm["quartile_coords"][1][1]) if len(worm["quartile_coords"]) > 1 else None,
                            "y_mid": int(worm["quartile_coords"][2][0]) if len(worm["quartile_coords"]) > 2 else None,
                            "x_mid": int(worm["quartile_coords"][2][1]) if len(worm["quartile_coords"]) > 2 else None,
                            "y_hip": int(worm["quartile_coords"][3][0]) if len(worm["quartile_coords"]) > 3 else None,
                            "x_hip": int(worm["quartile_coords"][3][1]) if len(worm["quartile_coords"]) > 3 else None,
                            "y_tail": int(worm["quartile_coords"][4][0]) if len(worm["quartile_coords"]) > 4 else None,
                            "x_tail": int(worm["quartile_coords"][4][1]) if len(worm["quartile_coords"]) > 4 else None,
                            "path_length": worm["path_length"]
                        }
                    writer.writerow(data)
