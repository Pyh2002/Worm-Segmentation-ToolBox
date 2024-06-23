import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.animation as animation
from matplotlib.cm import ScalarMappable


def create_animation():
    raw_data = pd.read_csv('raw_data.csv')
    processed_data = pd.read_csv('processed_data.csv')

    worm_data = raw_data[raw_data['worm_id'] == 0]

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
        x_min, x_max = worm_data['x_mid'].min(
        ) - 50, worm_data['x_mid'].max() + 50
        y_min, y_max = worm_data['y_mid'].min(
        ) - 50, worm_data['y_mid'].max() + 50
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
        ax.set_title('Mid position')
        ax.set_aspect('equal')
        ax.tick_params(axis='both', which='major', labelsize=20)
        if num == len(worm_data['x_mid']) - 1:
            plt.savefig('mid_position_graph.png')
        return scatter,

    ani = animation.FuncAnimation(fig, update_mid, frames=range(
        0, len(worm_data['x_mid'])), interval=10, blit=True)

    ani.save('animation_mid.mp4', writer='ffmpeg', fps=14.225)

    def update_head(num):
        x_min, x_max = worm_data['x_head'].min(
        ) - 50, worm_data['x_head'].max() + 50
        y_min, y_max = worm_data['y_head'].min(
        ) - 50, worm_data['y_head'].max() + 50
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
        ax.set_title('Head position')
        ax.set_aspect('equal')
        ax.tick_params(axis='both', which='major', labelsize=20)
        if num == len(worm_data['x_head']) - 1:
            plt.savefig('head_position_graph.png')
        return scatter,

    ani = animation.FuncAnimation(fig, update_head, frames=range(
        0, len(worm_data['x_head'])), interval=10, blit=True)

    ani.save('animation_head.mp4', writer='ffmpeg', fps=14.225)