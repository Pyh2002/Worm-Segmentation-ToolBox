import pandas as pd
import numpy as np
import cv2
import os


def switch_head_tail(row):
    row[['x_head', 'y_head', 'x_tail', 'y_tail']] = row[[
        'x_tail', 'y_tail', 'x_head', 'y_head']].values
    row[['x_neck', 'y_neck', 'x_hip', 'y_hip']] = row[[
        'x_hip', 'y_hip', 'x_neck', 'y_neck']].values
    return row


def process_intervals(path_to_images, raw_data_path, intervals_path, mode='auto'):
    raw_data = pd.read_csv(raw_data_path)
    intervals = pd.read_csv(intervals_path)

    for _, interval in intervals.iterrows():
        start_frame = interval['start_frame']
        end_frame = interval['end_frame']

        interval_data = raw_data[(raw_data['frame_number'] >= start_frame) & (
            raw_data['frame_number'] <= end_frame)]

        if mode == 'manual' and interval['status'] == 'Regular':
            first_frame = interval_data.iloc[0]
            frame_number_str = f'{int(first_frame["frame_number"]):04d}'
            print(frame_number_str, path_to_images)
            img_path = f'{path_to_images}/frame_{frame_number_str}_mask.png'
            img = cv2.imread(img_path)
            cv2.imshow('Select Head', img)
            head_point = cv2.selectROI(
                'Select Head', img, fromCenter=False, showCrosshair=True)
            cv2.destroyAllWindows()
            head_x, head_y = head_point[0], head_point[1]

            distances = np.sqrt(
                (interval_data['x_head'] - head_x)**2 + (interval_data['y_head'] - head_y)**2)
            if distances.iloc[0] > distances.iloc[1]:
                interval_data.iloc[0] = switch_head_tail(interval_data.iloc[0])
        for i in range(1, len(interval_data)):
            # Find the most recent frame with 'Regular' status and 'Normal' worm_status
            j = i - 1
            while j >= 0 and (interval_data.iloc[j]['status'] != 'Regular' or interval_data.iloc[j]['worm_status'] == 'Coiling or Splitted'):
                j -= 1

            if j >= 0:
                prev_row = interval_data.iloc[j]
            else:
                # Fallback to the immediate previous frame if no valid frame is found, should not happen in regular cases
                # If it does happen, break the loop and continue with the next interval
                break

            curr_row = interval_data.iloc[i]

            prev_head = np.array([prev_row['x_head'], prev_row['y_head']])
            curr_head = np.array([curr_row['x_head'], curr_row['y_head']])
            curr_tail = np.array([curr_row['x_tail'], curr_row['y_tail']])

            if np.linalg.norm(curr_head - prev_head) > np.linalg.norm(curr_tail - prev_head):
                interval_data.iloc[i] = switch_head_tail(curr_row)

        raw_data.update(interval_data)

    raw_data.to_csv('modified_raw_data.csv', index=False)


if __name__ == "__main__":
    import sys
    if len(sys.argv) != 5:
        print("Usage: python process_intervals.py <path_to_images> <raw_data_path> <intervals_path> <mode>")
        sys.exit(1)

    path_to_images = sys.argv[1]
    raw_data_path = sys.argv[2]
    intervals_path = sys.argv[3]
    mode = sys.argv[4]

    process_intervals(path_to_images, raw_data_path, intervals_path, mode)
