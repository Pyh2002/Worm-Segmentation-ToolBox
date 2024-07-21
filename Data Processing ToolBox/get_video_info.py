import os
import pandas as pd


def get_video_info(video_name, subfolder_path):
    processed_data = pd.read_csv(os.path.join(
        subfolder_path, 'processed_data.csv'))
    raw_data = pd.read_csv(os.path.join(subfolder_path, 'raw_data.csv'))
    video_info = {}
    fps = 14.225

    video_info_path = os.path.join(subfolder_path, 'video_info.csv')

    # Calculate video_info
    video_info['video_name'] = video_name
    video_info['total_frames'] = len(raw_data)
    video_info['total_time'] = len(raw_data) / fps
    video_info['effective_frame'] = len(raw_data[raw_data['worm_id']
                                                 == 1])
    video_info['forward_frames'] = len(processed_data[processed_data['direction']
                                                      == 'forward'])
    video_info['backward_frames'] = len(processed_data[processed_data['direction']
                                                       == 'backward'])
    video_info['still_frames'] = len(processed_data[processed_data['direction']
                                                    == 'still'])
    video_info['forward_prop'] = video_info['forward_frames'] / \
        video_info['effective_frame']
    video_info['backward_prop'] = video_info['backward_frames'] / \
        video_info['effective_frame']
    video_info['still_prop'] = video_info['still_frames'] / \
        video_info['effective_frame']
    video_info['possible_coling_worm'] = len(raw_data[raw_data['worm_id']
                                                      == 0])
    video_info['possible_mult_worms'] = len(raw_data[raw_data['worm_id']
                                                     == -1])
    video_info['average_1frame_speed'] = processed_data['speed_1frame'].mean()
    video_info['average_1second_speed'] = processed_data['speed_1second'].mean()
    video_info['average_10second_speed'] = processed_data['speed_10second'].mean()
    video_info['median_1frame_speed'] = processed_data['speed_1frame'].median()
    video_info['median_1second_speed'] = processed_data['speed_1second'].median()
    video_info['median_10second_speed'] = processed_data['speed_10second'].median()
    video_info['max_1frame_speed'] = processed_data['speed_1frame'].max()
    video_info['max_1second_speed'] = processed_data['speed_1second'].max()
    video_info['max_10second_speed'] = processed_data['speed_10second'].max()
    video_info['min_1frame_speed'] = processed_data['speed_1frame'].min()
    video_info['min_1second_speed'] = processed_data['speed_1second'].min()
    video_info['min_10second_speed'] = processed_data['speed_10second'].min()

    video_info_df = pd.DataFrame(
        list(video_info.items()), columns=['Variable', 'Value'])
    video_info_df = pd.DataFrame([video_info])
    video_info_df.to_csv(video_info_path, index=False)

    return video_info
