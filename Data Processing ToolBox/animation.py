import cv2
import os
import numpy as np
import pandas
import matplotlib.pyplot as plt
import matplotlib.animation as animation
import scipy

data = pandas.read_csv('data.csv')

worm_data = data[data['worm_id'] == 1]

fig, ax = plt.subplots()


def update_mid(num):
    ax.clear()
    colors = plt.cm.Reds(np.linspace(0, 1, num))
    scatter = ax.scatter(worm_data['x_mid'][:num], worm_data['y_mid'][:num], c=colors, s=2)
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
