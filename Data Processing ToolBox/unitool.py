import cv2
import os
import sys
import numpy
import csv
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.animation as animation
from scipy.signal import convolve2d
from PIL import Image, ImageDraw
from skimage.morphology import skeletonize
from skimage import data
from skimage.graph import route_through_array
from skimage.measure import label, regionprops


def extract_frames(video_name, extension):
    video_path = video_name + extension
    output_dir = video_name + "_frames"
    os.makedirs(output_dir, exist_ok=True)

    cap = cv2.VideoCapture(video_path)

    frame_count = 0
    while True:
        ret, frame = cap.read()
        if not ret:
            break

        print(f"Writing frame {frame_count:04d}")
        frame_filename = f"{output_dir}/frame_{frame_count:04d}.png"
        cv2.imwrite(frame_filename, frame)
        frame_count += 1

    cap.release()


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


def skeletonize_image(image):
    skeleton = skeletonize(image)
    return skeleton


def skeletonize_folder(input_folder_path, output_folder_path):
    sorted_file_names = sorted(os.listdir(input_folder_path))
    for file_name in sorted_file_names:
        print(file_name + " skeletonizing")
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

    for region in regionprops(labeled_mask):
        worm_mask = labeled_mask == region.label
        endpoint_coords, quartile_coords, path_length, path = analyze_skeleton(
            worm_mask)
        if path_length == 0:
            continue

        for y, x in endpoint_coords:
            draw.ellipse((x - 2, y - 2, x + 2, y + 2), outline="red", width=1)
        for y, x in quartile_coords:
            draw.ellipse((x - 2, y - 2, x + 2, y + 2),
                         fill="blue", outline="blue", width=1)

        worm_data.append({
            "endpoints": endpoint_coords,
            "quartile_coords": quartile_coords,
            "path_length": path_length
        })

    return worm_data, img


def create_endpoints_folder(input_folder_path, output_folder_path):
    sorted_file_names = sorted(os.listdir(input_folder_path))

    fieldnames = ['frame_number', 'worm_id', 'y_head', 'x_head', 'y_neck',
                  'x_neck', 'y_mid', 'x_mid', 'y_hip', 'x_hip', 'y_tail', 'x_tail', 'path_length']
    with open('raw_data.csv', 'w', newline='') as file:
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
                for worm_id, worm in enumerate(worm_data):
                    data = {
                        "frame_number": frame_number,
                        "worm_id": worm_id,
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


def calculate_eccentricity_angle(contours):
    # Create a list to store the data
    data = []

    for contour in contours:
        for single_contour in contour:
            moments = cv2.moments(single_contour)
            mu11 = moments["mu11"]
            mu02 = moments["mu02"]
            mu20 = moments["mu20"]

            a1 = (mu20 + mu02) / 2
            a2 = np.sqrt(4 * mu11**2 + (mu20 - mu02)**2) / 2

            minor_axis = a1 - a2
            major_axis = a1 + a2

            ecc = np.sqrt(1 - minor_axis / major_axis)
            ang = np.arctan2(2 * mu11, (mu20 - mu02)) / 2
            ang *= 180 / np.pi

            # Add the data to the list
            data.append({'eccentricity': ecc, 'angle': ang})

    # Write the data to the CSV file
    fieldnames = ['eccentricity', 'angle']
    with open('processed_data.csv', 'w', newline='') as file:
        writer = csv.DictWriter(file, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(data)


def overlay_images(base_image_path, overlay_image_path, output_image_path):
    # Load images
    base_image = Image.open(base_image_path).convert("RGBA")
    overlay_image = Image.open(overlay_image_path).convert("RGBA")
    if base_image.size != overlay_image.size:
        overlay_image = overlay_image.resize(base_image.size, Image.ANTIALIAS)

    r, g, b, a = overlay_image.split()
    r = np.array(r)
    g = np.array(g)
    b = np.array(b)
    # Remove all the black part
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


def images_to_video(input_folder_path, output_video_path, fps=14.225):
    images = [img for img in sorted(os.listdir(
        input_folder_path)) if img.endswith((".png", ".jpg", ".jpeg"))]
    sample_img_path = os.path.join(input_folder_path, images[0])
    frame = cv2.imread(sample_img_path)
    height, width, layers = frame.shape

    fourcc = cv2.VideoWriter_fourcc(*'mp4v')
    video = cv2.VideoWriter(output_video_path, fourcc, fps, (width, height))
    for image in images:
        img_path = os.path.join(input_folder_path, image)
        frame = cv2.imread(img_path)
        video.write(frame)
    video.release()


def calculate_speed():
    data = pd.read_csv('raw_data.csv')
    processed_data = pd.read_csv('processed_data.csv')
    processed_data['speed_1frame'] = numpy.sqrt(
        (data['x_mid'] - data['x_mid'].shift())**2 + (data['y_mid'] - data['y_mid'].shift())**2)
    processed_data['speed_1second'] = numpy.sqrt(
        (data['x_mid'] - data['x_mid'].shift(14))**2 + (data['y_mid'] - data['y_mid'].shift(14))**2)
    processed_data['speed_10second'] = numpy.sqrt(
        (data['x_mid'] - data['x_mid'].shift(140))**2 + (data['y_mid'] - data['y_mid'].shift(142))**2)/10

    processed_data.to_csv('processed_data.csv', index=False)


def create_animation():
    data = pd.read_csv('raw_data.csv')

    worm_data = data[data['worm_id'] == 1]

    fig, ax = plt.subplots()

    def update_mid(num):
        ax.clear()
        colors = plt.cm.Reds(np.linspace(0, 1, num))
        scatter = ax.scatter(
            worm_data['x_mid'][:num], worm_data['y_mid'][:num], c=colors, s=2)
        ax.text(0.02, 0.95, f'Frame: {num}', transform=ax.transAxes, va='top')
        return scatter,

    ani = animation.FuncAnimation(fig, update_mid, frames=range(
        0, len(worm_data['x_mid'])), interval=10, blit=True)

    ani.save('animation_mid.mp4', writer='ffmpeg', fps=14.225)

    def update_head(num):
        ax.clear()
        line, = ax.plot(worm_data['x_head'][:num], worm_data['y_head']
                        [:num], 'ro', markersize=2)
        ax.text(0.02, 0.95, f'Frame: {num}', transform=ax.transAxes, va='top')
        return line,

    ani = animation.FuncAnimation(fig, update_head, frames=range(
        0, len(worm_data['x_head'])), interval=10, blit=True)

    ani.save('animation_head.mp4', writer='ffmpeg', fps=14.225)


if __name__ == "__main__":
    video = sys.argv[1]
    video = os.path.normpath(video)
    video_name, extension = os.path.splitext(video)
    print(video_name, extension)

    # extract_frames(video_name, extension)

    contours = get_contours_folder(video_name + "_frames",
                                   video_name + "_contours")

    # skeletonize_folder(video_name + "_frames",
    #                    video_name + "_skeletonized_masks")
    # create_endpoints_folder(
    #     video_name + "_skeletonized_masks", video_name + "_endpoints")

    # overlay_folders(video_name + "_frames",
    #                 video_name + "_endpoints", video_name + "_overlayed_images")
    # images_to_video(video_name + '_overlayed_images',
    #                 video_name + '_output_video.mp4')

    calculate_eccentricity_angle(contours)
    calculate_speed()
    create_animation()
