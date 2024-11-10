import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import os


def calculate_movement(start_frame, end_frame, parentfolder_path, processed_data, worm_id, threshold=0.5):
    raw_data = pd.read_csv(os.path.join(
        parentfolder_path, 'modified_raw_data.csv'))

    begin_index = raw_data[raw_data['frame_number'] == start_frame].index[0]
    end_index = raw_data[raw_data['frame_number'] == end_frame].index[-1]

    interval_data = raw_data.loc[begin_index:end_index]
    interval_data = interval_data[interval_data['worm_status']
                                  != 'Coiling or Splitted']
    interval_data = interval_data[interval_data['worm_id'] == worm_id]

    def calculate_group_movement(group):
        group['neck_head_direction'] = list(zip(
            group['x_head'] - group['x_neck'], group['y_head'] - group['y_neck']))
        group['0.5sec_movement'] = list(zip(
            group['x_mid'] - group['x_mid'].shift(7), group['y_mid'] - group['y_mid'].shift(7)))

        def calculate_dot_product(v1, v2):
            return v1[0]*v2[0] + v1[1]*v2[1]

        def calculate_norm(v):
            return np.sqrt(v[0]**2 + v[1]**2)

        group['0.5sec_movement_norm'] = group['0.5sec_movement'].apply(
            calculate_norm)

        movement_threshold = group['0.5sec_movement_norm'].mean() * threshold

        def calculate_direction_change(row):
            neck_head_norm = calculate_norm(row['neck_head_direction'])
            movement_norm = calculate_norm(row['0.5sec_movement'])
            if neck_head_norm == 0 or movement_norm == 0:
                return 0
            return np.arccos(
                calculate_dot_product(row['neck_head_direction'], row['0.5sec_movement']) / (
                    neck_head_norm * movement_norm
                ))

        group['direction_change'] = group.apply(lambda row: calculate_direction_change(
            row) if row['0.5sec_movement_norm'] >= movement_threshold else 0, axis=1)

        group['direction'] = np.where(
            group['0.5sec_movement_norm'] < movement_threshold, 'still',
            np.where(group['direction_change'] > np.pi/2, 'backward', 'forward'))

        return group

    movement_data = interval_data.groupby('worm_id').apply(
        calculate_group_movement).reset_index(drop=True)

    # Merge the movement data back into the processed_data
    processed_data = processed_data.merge(
        movement_data[['frame_number', 'worm_id', 'neck_head_direction', '0.5sec_movement',
                       '0.5sec_movement_norm', 'direction_change', 'direction']],
        on=['frame_number', 'worm_id'],
        how='left'
    )

    return processed_data


def create_movement_graph(idx, start_frame, end_frame, parentfolder_path, video_name, worm_id):
    worm_data = pd.read_csv(os.path.join(
        parentfolder_path, 'modified_raw_data.csv'))
    processed_data = pd.read_csv(os.path.join(
        parentfolder_path, f"processed_data_{idx}.csv"))

    merged_data = pd.merge(worm_data, processed_data, on=[
                           'frame_number', 'worm_id'], how='inner')
    merged_data = merged_data[merged_data['worm_id'] == worm_id]
    merged_data = merged_data[merged_data['worm_status']
                              != 'Coiling or Splitted']
    plt.figure(figsize=(10, 8))
    scatter = plt.scatter(merged_data['x_mid'], merged_data['y_mid'],
                          c=merged_data['direction'].map(
        {'forward': 'red', 'backward': 'blue', 'still': 'green'}),
        s=merged_data['0.5sec_movement_norm']*10, alpha=0.25)

    # Add a colorbar
    plt.colorbar(scatter, label='Movement Direction')
    plt.xlabel('X position')
    plt.ylabel('Y position')
    plt.title(video_name + ' Movement')
    plt.gca().invert_yaxis()
    plt.gca().set_aspect('equal')

    colors = {'forward': 'red', 'backward': 'blue', 'still': 'green'}
    labels = list(colors.keys())
    handles = [plt.Rectangle((0, 0), 1, 1, color=colors[label], alpha=0.25)
               for label in labels]
    plt.legend(handles, labels, title="Movement Direction")

    plt.savefig(os.path.join(parentfolder_path, f'movement_graph_{idx}.png'))
    plt.close()
