import numpy
import pandas as pd


def calculate_speed():
    data = pd.read_csv('raw_data.csv')
    processed_data = pd.read_csv('processed_data.csv')
    processed_data['speed_1frame'] = numpy.sqrt(
        (data['x_mid'] - data['x_mid'].shift())**2 + (data['y_mid'] - data['y_mid'].shift())**2)
    processed_data['speed_1second'] = numpy.sqrt(
        (data['x_mid'] - data['x_mid'].shift(14))**2 + (data['y_mid'] - data['y_mid'].shift(14))**2)
    processed_data['speed_10second'] = numpy.sqrt(
        (data['x_mid'] - data['x_mid'].shift(140))**2 + (data['y_mid'] - data['y_mid'].shift(142))**2)/10

    processed_data.to_csv('processed_data.csv', index=False)
