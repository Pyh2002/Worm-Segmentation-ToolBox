import os
import sys

from extract_frames import extract_frames
from extract_contours import get_contours_folder
from extract_skeletons import skeletonize_folder
from generate_endpoints import create_endpoints_folder
from generate_overlays import overlay_folders
from make_video import images_to_video

from calc_speed import calculate_speed
from calc_eccentricity import calculate_eccentricity_angle
from calc_movement import calculate_movement
from calc_movement import create_movement_graph
from create_animation import create_animation


if __name__ == "__main__":
    video = sys.argv[1]
    video = os.path.normpath(video)
    video_name, extension = os.path.splitext(video)
    print(video_name, extension)

    extract_frames(video_name, extension)

    contours = get_contours_folder(video_name + "_frames",
                                   video_name + "_contours")

    skeletonize_folder(video_name + "_frames",
                       video_name + "_skeletonized_masks")
    create_endpoints_folder(
        video_name + "_skeletonized_masks", video_name + "_endpoints")

    overlay_folders(video_name + "_frames",
                    video_name + "_endpoints", video_name + "_overlayed_images")
    images_to_video(video_name + '_overlayed_images',
                    video_name + '_output_video.mp4')

    calculate_speed()
    calculate_eccentricity_angle(contours)
    calculate_movement()
    create_animation()
    create_movement_graph()
