import numpy as np
import pandas as pd
from scipy.signal import convolve2d
from skimage.graph import route_through_array
from skimage.measure import label, regionprops
import cv2
import os


def analyze_skeleton(region):
    kernel = np.array([[1, 1, 1], [1, 0, 1], [1, 1, 1]])
    neighbor_count = convolve2d(
        region, kernel, mode="same", boundary="fill", fillvalue=0)

    endpoints = (neighbor_count == 1) & region
    endpoint_coords = np.argwhere(endpoints)
    if len(endpoint_coords) == 2:
        path, _ = route_through_array(
            1 - region,
            start=endpoint_coords[0],
            end=endpoint_coords[-1],
            fully_connected=True,
        )
        path = np.array(path)
        status = 'Normal'
    else:
        status = 'Coiling or Splitted'

    if status == 'Normal':
        path_length = len(path)
        quartile_indices = np.linspace(0, len(path) - 1, 5, dtype=int)
        quartile_coords = [path[index]
                           for index in quartile_indices if index < len(path)]
        return status, quartile_coords, path_length
    else:
        return status, [], 0


def create_endpoints(df, frame_number, image):
    labeled_img = label(image, connectivity=2)
    regions = regionprops(labeled_img)

    if len(regions) == 0:
        status = 'Empty'
    elif len(regions) == 1:
        status = 'Regular'
    elif len(regions) == 2:
        status = 'Multiple'
    else:
        status = 'Disrupted'

    if status == 'Empty' or status == 'Disrupted':
        new_row = {'frame_number': frame_number, 'status': status, 'worm_id': 0, 'y_head': 0, 'x_head': 0,
                   'y_neck': 0, 'x_neck': 0, 'y_mid': 0, 'x_mid': 0, 'y_hip': 0, 'x_hip': 0, 'y_tail': 0, 'x_tail': 0, 'path_length': 0, 'worm_status': 'NAN'}
        new_row_df = pd.DataFrame([new_row])
        df = pd.concat([df, new_row_df], ignore_index=True)

    for region in regions:
        region_mask = (labeled_img == region.label)
        # cv2.imwrite('region_mask.png', region_mask.astype(np.uint8) * 255)
        if status == 'Regular' or status == 'Multiple':
            worm_status, quartile_coords, path_length = analyze_skeleton(
                region_mask)
            if worm_status == 'Normal':
                new_row = {'frame_number': frame_number, 'status': status, 'worm_id': region.label, 'y_head': quartile_coords[0][0], 'x_head': quartile_coords[0][1], 'y_neck': quartile_coords[1][0], 'x_neck': quartile_coords[1][
                    1], 'y_mid': quartile_coords[2][0], 'x_mid': quartile_coords[2][1], 'y_hip': quartile_coords[3][0], 'x_hip': quartile_coords[3][1], 'y_tail': quartile_coords[4][0], 'x_tail': quartile_coords[4][1], 'path_length': path_length, 'worm_status': worm_status}
            elif worm_status == 'Coiling or Splitted':
                new_row = {'frame_number': frame_number, 'status': status, 'worm_id': region.label, 'y_head': 0, 'x_head': 0,
                           'y_neck': 0, 'x_neck': 0, 'y_mid': 0, 'x_mid': 0, 'y_hip': 0, 'x_hip': 0, 'y_tail': 0, 'x_tail': 0, 'path_length': 0, 'worm_status': worm_status}
            new_row_df = pd.DataFrame([new_row])
            df = pd.concat([df, new_row_df], ignore_index=True)
    return df


def create_endpoints_folder(parentfolder_path, input_folder_path):
    sorted_input_file_names = sorted(os.listdir(input_folder_path))
    raw_data_path = os.path.join(parentfolder_path, "raw_data.csv")

    fieldnames = ['frame_number', 'status', 'worm_id', 'y_head', 'x_head', 'y_neck',
                  'x_neck', 'y_mid', 'x_mid', 'y_hip', 'x_hip', 'y_tail', 'x_tail', 'path_length', 'worm_status']
    df = pd.DataFrame(columns=fieldnames)
    for file_name in sorted_input_file_names:
        print(file_name, 'generating endpoints')
        if file_name.endswith(".png"):
            frame_number = int(file_name.split('_')[1])
            input_image_path = os.path.join(input_folder_path, file_name)
            img = cv2.imread(input_image_path, cv2.IMREAD_GRAYSCALE)
            df = create_endpoints(df, frame_number, img)
    df.to_csv(raw_data_path, index=False)


if __name__ == "__main__":
    import sys
    if len(sys.argv) != 3:
        print("Usage: python generate_endpoints.py <parentfolder_path> <input_folder_path>")
        sys.exit(1)
    create_endpoints_folder(sys.argv[1], sys.argv[2])
