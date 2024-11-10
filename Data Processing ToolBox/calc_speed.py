import os
import numpy as np
import pandas as pd


def calculate_speed(start_frame, end_frame, parentfolder_path, processed_data, worm_id):
    raw_data_path = os.path.join(parentfolder_path, "modified_raw_data.csv")
    raw_data = pd.read_csv(raw_data_path)

    begin_index = raw_data[raw_data['frame_number'] == start_frame].index[0]
    end_index = raw_data[raw_data['frame_number'] == end_frame].index[-1]

    interval_data = raw_data.loc[begin_index:end_index]
    interval_data = interval_data[interval_data['worm_status'] != 'Coiling or Splitted']
    interval_data = interval_data[interval_data['worm_id'] == worm_id]

    # Group by worm_id and calculate speed within each group
    def calculate_group_speed(group):
        group['speed_1frame'] = np.sqrt(
            (group['x_mid'] - group['x_mid'].shift(1))**2 + (group['y_mid'] - group['y_mid'].shift(1))**2)
        group['speed_1second'] = np.sqrt(
            (group['x_mid'] - group['x_mid'].shift(14))**2 + (group['y_mid'] - group['y_mid'].shift(14))**2)
        group['speed_10second'] = np.sqrt(
            (group['x_mid'] - group['x_mid'].shift(140))**2 + (group['y_mid'] - group['y_mid'].shift(140))**2)/10
        return group

    speed_data = interval_data.groupby('worm_id').apply(
        calculate_group_speed).reset_index(drop=True)

    # Merge the speed data back into the processed_data
    processed_data = processed_data.merge(
        speed_data[['frame_number', 'worm_id', 'speed_1frame',
                    'speed_1second', 'speed_10second']],
        on=['frame_number', 'worm_id'],
        how='left'
    )

    return processed_data
