import os
import numpy as np
import pandas as pd
from PIL import Image
from PIL import ImageDraw


def overlay_images(frame_number, base_image_path, output_image_path, parentfolder_path='./'):
    base_image = Image.open(base_image_path).convert("RGBA")
    raw_data = pd.read_csv(os.path.join(
        parentfolder_path, 'modified_raw_data.csv'))
    frame_data = raw_data[raw_data['frame_number'] == frame_number]
    overlay_image = Image.new('RGBA', base_image.size, (0, 0, 0, 0))
    draw = ImageDraw.Draw(overlay_image)
    for index, row in frame_data.iterrows():
        if row['status'] == 'Regular' or row['status'] == 'Multiple':
            # Draw circles
            draw.ellipse((row['x_head']-10, row['y_head']-10,
                          row['x_head']+10, row['y_head']+10), fill='red')
            draw.ellipse((row['x_neck']-4, row['y_neck']-4,
                         row['x_neck']+4, row['y_neck']+4), fill='red')
            draw.ellipse((row['x_mid']-4, row['y_mid']-4,
                         row['x_mid']+4, row['y_mid']+4), fill='red')
            draw.ellipse((row['x_hip']-4, row['y_hip']-4,
                         row['x_hip']+4, row['y_hip']+4), fill='red')
            draw.ellipse((row['x_tail']-4, row['y_tail']-4,
                         row['x_tail']+4, row['y_tail']+4), fill='red')

            # Draw lines
            draw.line([(row['x_head'], row['y_head']),
                      (row['x_neck'], row['y_neck'])], fill='black', width=2)
            draw.line([(row['x_neck'], row['y_neck']),
                      (row['x_mid'], row['y_mid'])], fill='black', width=2)
            draw.line([(row['x_mid'], row['y_mid']),
                      (row['x_hip'], row['y_hip'])], fill='black', width=2)
            draw.line([(row['x_hip'], row['y_hip']),
                      (row['x_tail'], row['y_tail'])], fill='black', width=2)

    combined = Image.alpha_composite(base_image, overlay_image)
    combined.save(output_image_path)


def overlay_folders(parentfolder_path, input_folder_path, output_folder_path):
    sorted_base_file_names = sorted(os.listdir(input_folder_path))
    for base_file_name in sorted_base_file_names:
        if base_file_name.endswith(".png"):
            frame_number = int(base_file_name.split('_')[1])
            print(base_file_name + " overlaying")
            if not os.path.exists(output_folder_path):
                os.makedirs(output_folder_path)
            base_image_path = os.path.join(input_folder_path, base_file_name)
            output_image_path = os.path.join(
                output_folder_path, base_file_name)
            overlay_images(frame_number, base_image_path,
                           output_image_path, parentfolder_path)


if __name__ == "__main__":
    import sys
    overlay_folders("./", sys.argv[1], sys.argv[2])
