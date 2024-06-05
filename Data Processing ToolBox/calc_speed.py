import cv2
import os
import numpy
import pandas
import matplotlib.pyplot as plt
import matplotlib.animation as animation
import scipy
import numpy as np

video_name = "WormParticles45um-3.5-Wrm1E_frames"
data = pandas.read_csv('data.csv')
data['speed_1frame'] = numpy.sqrt(
    (data['x_mid'] - data['x_mid'].shift())**2 + (data['y_mid'] - data['y_mid'].shift())**2)
data['speed_1second'] = numpy.sqrt(
    (data['x_mid'] - data['x_mid'].shift(14))**2 + (data['y_mid'] - data['y_mid'].shift(14))**2)
data['speed_10second'] = numpy.sqrt(
    (data['x_mid'] - data['x_mid'].shift(140))**2 + (data['y_mid'] - data['y_mid'].shift(142))**2)/10

data['angle'] = numpy.arctan2(
    data['y_tail'] - data['y_head'], data['x_tail'] - data['x_head'])
data['angular_speed_1frame'] = numpy.abs(data['angle'] - data['angle'].shift())
data['angular_speed_1second'] = numpy.abs(
    data['angle'] - data['angle'].shift(14))
data['angular_speed_10second'] = numpy.abs(
    data['angle'] - data['angle'].shift(142))/10

data.to_csv('data.csv', index=False)

fig, ax = plt.subplots()
x = numpy.arange(0, len(data), 1)
y = data['speed_1second']
line, = ax.plot(x, y)
ax.set_ylim(0, 100)
ax.set_xlim(0, len(data))
ax.set_title('Speed of the object')
ax.set_xlabel('Frame')
ax.set_ylabel('Speed (pixel/second)')

fig, ax = plt.subplots()
x = numpy.arange(0, len(data), 1)
y = data['speed_1frame']
line, = ax.plot(x, y)
ax.set_ylim(0, 100)
ax.set_xlim(0, len(data))
ax.set_title('Speed of the object')
ax.set_xlabel('Frame')
ax.set_ylabel('Speed (pixel/frame)')

fig, ax = plt.subplots()
x = numpy.arange(0, len(data), 1)
y = data['speed_10second']
line, = ax.plot(x, y)
ax.set_ylim(0, 100)
ax.set_xlim(0, len(data))
ax.set_title('Average speed of the object in 10 seconds')
ax.set_xlabel('Frame')
ax.set_ylabel('Speed (pixel/second)')

fig, ax = plt.subplots()
x = numpy.arange(0, len(data), 1)
y = data['angular_speed_1second']
line, = ax.plot(x, y)
ax.set_ylim(0, 0.5)
ax.set_xlim(0, len(data))
ax.set_title('Angular speed of the object')
ax.set_xlabel('Frame')
ax.set_ylabel('Angular speed (radian/second)')

fig, ax = plt.subplots()
x = numpy.arange(0, len(data), 1)
y = data['angular_speed_1frame']
line, = ax.plot(x, y)
ax.set_ylim(0, 0.2)
ax.set_xlim(0, len(data))
ax.set_title('Angular speed of the object')
ax.set_xlabel('Frame')
ax.set_ylabel('Angular speed (radian/frame)')

fig, ax = plt.subplots()
x = numpy.arange(0, len(data), 1)
y = data['angular_speed_10second']
line, = ax.plot(x, y)
ax.set_ylim(0, 0.2)
ax.set_xlim(0, len(data))
ax.set_title('Average angular speed of the object in 10 seconds')
ax.set_xlabel('Frame')
ax.set_ylabel('Angular speed (radian/frame)')
plt.show()
