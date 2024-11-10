import os
import pandas as pd


def get_video_info(video_name, parentfolder_path, index):
    processed_data = pd.read_csv(os.path.join(
        parentfolder_path, f'processed_data_{index}.csv'))
    raw_data = pd.read_csv(os.path.join(
        parentfolder_path, 'modified_raw_data.csv'))
    interval_data = pd.read_csv(os.path.join(
        parentfolder_path, 'intervals.csv'))
    start_frame = interval_data['start_frame'].iloc[index]
    end_frame = interval_data['end_frame'].iloc[index]
    raw_data = raw_data[(raw_data['frame_number'] >= start_frame) &
                        (raw_data['frame_number'] <= end_frame)]
    video_info = {}
    fps = 14.225

    video_info_path = os.path.join(
        parentfolder_path, f'video_info_{index}.csv')

    video_info['video_name'] = video_name
    video_info['total_frames'] = interval_data['num_frames'].iloc[index]
    video_info['total_time'] = interval_data['num_frames'].iloc[index] / fps
    video_info['effective_frames'] = len(raw_data[(raw_data['status'] == 'Regular') |
                                                  (raw_data['status'] == 'Multiple')])
    video_info['forward_frames'] = len(processed_data[processed_data['direction']
                                                      == 'forward'])
    video_info['backward_frames'] = len(processed_data[processed_data['direction']
                                                       == 'backward'])
    video_info['still_frames'] = len(processed_data[processed_data['direction']
                                                    == 'still'])
    video_info['forward_prop'] = video_info['forward_frames'] / \
        video_info['effective_frames']
    video_info['backward_prop'] = video_info['backward_frames'] / \
        video_info['effective_frames']
    video_info['still_prop'] = video_info['still_frames'] / \
        video_info['effective_frames']
    video_info['single_worm_frames'] = len(
        raw_data[raw_data['status'] == 'Regular'])
    video_info['mult_worms_frames'] = len(
        raw_data[raw_data['status'] == 'Multiple'])
    video_info['empty_frames'] = len(raw_data[raw_data['status'] == 'Empty'])
    video_info['disrupted_frames'] = len(
        raw_data[raw_data['status'] == 'Disrupted'])
    video_info['coiling_or_split_frames'] = len(
        raw_data[raw_data['status'] == 'Coiling or Splitted'])
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
