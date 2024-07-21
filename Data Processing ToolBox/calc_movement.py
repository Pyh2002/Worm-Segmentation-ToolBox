import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import os


def calculate_movement(subfolder_path):
    data = pd.read_csv(os.path.join(subfolder_path, 'raw_data.csv'))
    processed_data = pd.read_csv(os.path.join(
        subfolder_path, 'processed_data.csv'))

    processed_data['neck_head_direction'] = list(zip(
        data['x_head'] - data['x_neck'], data['y_head'] - data['y_neck']))
    processed_data['0.5sec_movement'] = list(zip(
        data['x_mid'] - data['x_mid'].shift(7), data['y_mid'] - data['y_mid'].shift(7)))

    def calculate_dot_product(v1, v2):
        return v1[0]*v2[0] + v1[1]*v2[1]

    def calculate_norm(v):
        return np.sqrt(v[0]**2 + v[1]**2)

    processed_data['0.5sec_movement_norm'] = processed_data['0.5sec_movement'].apply(
        calculate_norm)

    movement_threshold = processed_data['0.5sec_movement_norm'].mean()/2

    processed_data['direction_change'] = processed_data.apply(lambda row: np.arccos(
        calculate_dot_product(row['neck_head_direction'], row['0.5sec_movement']) / (
            calculate_norm(row['neck_head_direction']) *
            calculate_norm(row['0.5sec_movement'])
        )) if row['0.5sec_movement_norm'] >= movement_threshold else 0, axis=1)

    processed_data['direction'] = np.where(
        processed_data['0.5sec_movement_norm'] < movement_threshold, 'still',
        np.where(processed_data['direction_change'] > np.pi/2, 'backward', 'forward'))

    processed_data.to_csv(os.path.join(
        subfolder_path, 'processed_data.csv'), index=False)


def create_movement_graph(subfolder_path, video_name):
    worm_data = pd.read_csv(os.path.join(subfolder_path, 'raw_data.csv'))
    processed_data = pd.read_csv(os.path.join(
        subfolder_path, 'processed_data.csv'))

    plt.figure(figsize=(10, 8))

    scatter = plt.scatter(worm_data['x_mid'], worm_data['y_mid'],
                          c=processed_data['direction'].map(
        {'forward': 'red', 'backward': 'blue', 'still': 'green'}),
        s=processed_data['0.5sec_movement_norm']*10, alpha=0.25)
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

    plt.savefig(os.path.join(subfolder_path, 'movement_graph.png'))
