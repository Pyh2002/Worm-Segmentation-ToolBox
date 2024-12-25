import os
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.animation as animation
from matplotlib.cm import ScalarMappable


def generate_animation(idx, start_frame, end_frame, parentfolder_path, video_name):
    raw_data = pd.read_csv(os.path.join(
        parentfolder_path, 'modified_raw_data.csv'))
    processed_data = pd.read_csv(os.path.join(
        parentfolder_path, f"processed_data_{idx}.csv"))

    worm_data = raw_data[(raw_data['worm_id'] == 1) &
                         (raw_data['frame_number'] >= start_frame) &
                         (raw_data['frame_number'] <= end_frame)]

    dpi = 100
    fig_width = 1920 / dpi
    fig_height = 1440 / dpi
    fig, ax = plt.subplots(figsize=(fig_width, fig_height), dpi=dpi)

    norm = plt.Normalize(vmin=0, vmax=len(worm_data['x_mid']))
    sm = ScalarMappable(cmap='Reds', norm=norm)
    sm.set_array([])

    cbar = fig.colorbar(sm, ax=ax, orientation='vertical',
                        fraction=0.046, pad=0.04)
    cbar.set_label('Time progression')

    def update_mid(num):
        x_min = worm_data['x_mid'].dropna().min() - 50
        x_max = worm_data['x_mid'].dropna().max() + 50
        y_min = worm_data['y_mid'].dropna().min() - 50
        y_max = worm_data['y_mid'].dropna().max() + 50

        ax.clear()
        colors = plt.cm.Reds(np.linspace(0, 1, num))
        scatter = ax.scatter(
            worm_data['x_mid'][:num], worm_data['y_mid'][:num], c=colors, s=2)
        if num > 1:
            for i in range(1, num):
                ax.plot(worm_data['x_mid'][i-1:i+1],
                        worm_data['y_mid'][i-1:i+1], color=colors[i-1])
        ax.text(0.02, 0.95, f'Frame: {num}',
                transform=ax.transAxes, va='top', fontsize=28)
        ax.text(
            0.02, 0.85, f'Direction: {processed_data["direction"].iloc[num]}', transform=ax.transAxes, va='top', fontsize=28)
        ax.set_xlim(x_min, x_max)
        ax.set_ylim(y_min, y_max)
        ax.invert_yaxis()
        ax.set_xlabel('X position')
        ax.set_ylabel('Y position')
        ax.set_title(video_name + ' Mid position')
        ax.set_aspect('equal')
        ax.tick_params(axis='both', which='major', labelsize=20)
        if num == len(worm_data['x_mid']) - 1:
            plt.savefig(f'mid_position_graph_{idx}.png')
        return scatter,

    ani = animation.FuncAnimation(fig, update_mid, frames=range(
        0, len(worm_data['x_mid'])), interval=10, blit=True)

    ani.save(f'animation_mid_{idx}.mp4', writer='ffmpeg', fps=14.225)

    def update_head(num):
        x_min = worm_data['x_head'].dropna().min() - 50
        x_max = worm_data['x_head'].dropna().max() + 50
        y_min = worm_data['y_head'].dropna().min() - 50
        y_max = worm_data['y_head'].dropna().max() + 50

        ax.clear()
        colors = plt.cm.Reds(np.linspace(0, 1, num))
        scatter = ax.scatter(
            worm_data['x_head'][:num], worm_data['y_head'][:num], c=colors, s=2)
        if num > 1:
            for i in range(1, num):
                ax.plot(worm_data['x_head'][i-1:i+1],
                        worm_data['y_head'][i-1:i+1], color=colors[i-1])
        ax.text(0.02, 0.95, f'Frame: {num}',
                transform=ax.transAxes, va='top', fontsize=28)
        ax.text(
            0.02, 0.85, f'Direction: {processed_data["direction"].iloc[num]}', transform=ax.transAxes, va='top', fontsize=28)
        ax.set_xlim(x_min, x_max)
        ax.set_ylim(y_min, y_max)
        ax.invert_yaxis()
        ax.set_xlabel('X position')
        ax.set_ylabel('Y position')
        ax.set_title(video_name + ' Head position')
        ax.set_aspect('equal')
        ax.tick_params(axis='both', which='major', labelsize=20)
        if num == len(worm_data['x_head']) - 1:
            plt.savefig(f'head_position_graph_{idx}.png')
        return scatter,

    ani = animation.FuncAnimation(fig, update_head, frames=range(
        0, len(worm_data['x_head'])), interval=10, blit=True)

    ani.save(f'animation_head_{idx}.mp4', writer='ffmpeg', fps=14.225)


def generate_trace(idx, start_frame, end_frame, parentfolder_path, video_name, worm_id, color_map='viridis'):
    raw_data = pd.read_csv(os.path.join(parentfolder_path, 'raw_data.csv'))
    processed_data = pd.read_csv(os.path.join(
        parentfolder_path, f"processed_data_{idx}.csv"))
    worm_data = raw_data[(raw_data['worm_id'] == worm_id) &
                         (raw_data['frame_number'] >= start_frame) &
                         (raw_data['frame_number'] <= end_frame) &
                         (raw_data['worm_status'] != 'Coiling or Splitted')]
    num_points = len(worm_data['x_mid'])
    colors = plt.cm.get_cmap(color_map)(
        np.linspace(0, 1, len(worm_data['x_mid'])))
    fig, ax = plt.subplots()
    scatter = ax.scatter(
        worm_data['x_mid'][:num_points], worm_data['y_mid'][:num_points], c=colors, s=2)
    ax.set_xlabel('X position')
    ax.set_ylabel('Y position')
    ax.set_title(video_name + ' Mid trace ' + str(start_frame) + '-' + str(end_frame) + ': worm_id = ' + str(worm_id))
    ax.set_aspect('equal')
    ax.invert_yaxis()
    ax.tick_params(axis='both', which='major', labelsize=10)

    if num_points > 1:
        for i in range(1, num_points):
            ax.plot(worm_data['x_mid'][i-1:i+1],
                    worm_data['y_mid'][i-1:i+1], color=colors[i-1])

    cmap = plt.cm.get_cmap(color_map)

    sm = plt.cm.ScalarMappable(
        cmap=cmap, norm=plt.Normalize(vmin=0, vmax=num_points))
    sm.set_array([])
    cbar = fig.colorbar(sm, ax=ax, orientation='vertical',
                        fraction=0.046, pad=0.04)
    cbar.set_label('Time progression')
    plt.savefig(os.path.join(parentfolder_path, f'trace_graph_{idx}.png'))
