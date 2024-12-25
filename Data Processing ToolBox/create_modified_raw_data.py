import pandas as pd
import numpy as np
import cv2
import os


def switch_head_tail(row):
    # Create a copy of the row to avoid SettingWithCopyWarning
    row_copy = row.copy()
    row_copy.loc[['x_head', 'y_head', 'x_tail', 'y_tail']] = row_copy.loc[[
        'x_tail', 'y_tail', 'x_head', 'y_head']].values
    row_copy.loc[['x_neck', 'y_neck', 'x_hip', 'y_hip']] = row_copy.loc[[
        'x_hip', 'y_hip', 'x_neck', 'y_neck']].values
    return row_copy


def create_modified_raw_data(path_to_images, raw_data_path, intervals_path, contours_dict, mode='auto'):
    raw_data = pd.read_csv(raw_data_path)
    intervals = pd.read_csv(intervals_path)

    for _, interval in intervals.iterrows():
        # interval example:
        # start_frame,end_frame,num_frames,status,switch_status,worm_id
        start_frame = interval['start_frame']
        end_frame = interval['end_frame']
        worm_id = interval['worm_id']
        # print("start_frame", start_frame)
        # print("end_frame", end_frame)

        interval_data = raw_data[(raw_data['frame_number'] >= start_frame) & (
            raw_data['frame_number'] <= end_frame) & (raw_data['worm_id'] == worm_id)]

        # if interval['status'] == 'Regular' and mode == 'manual':
        if mode == 'manual':
            first_frame = interval_data.iloc[0]

            contours = contours_dict[start_frame]
            # print("contours", len(contours))
            contour = contours[worm_id - 1]

            # Create a blank mask with the same size as the image
            filled_contour = np.zeros(
                (1440, 1920), dtype=np.uint8)  # Create a black mask

            # Fill the contour area with white (255)
            cv2.drawContours(filled_contour, [contour], 0,
                             (255), thickness=cv2.FILLED)

            # Apply dilation to the filled contour with a kernel size that extends by 30 pixels
            # Adjusted kernel size for 30px dilation
            kernel = np.ones((30, 30), np.uint8)
            filled_contour = cv2.dilate(filled_contour, kernel, iterations=1)

            frame_number_str = f'{int(first_frame["frame_number"]):04d}'
            # print(frame_number_str, path_to_images)
            img_path = f'{path_to_images}/frame_{frame_number_str}.png'
            img = cv2.imread(img_path)

            # Resize the image to 1920x1440
            img = cv2.resize(img, (1920, 1440))
            # Mask the image with the dilated contour
            img = cv2.bitwise_and(img, img, mask=filled_contour)

            # Make the window resizable
            cv2.namedWindow('Select Head', cv2.WINDOW_NORMAL)
            cv2.imshow('Select Head', img)
            cv2.resizeWindow('Select Head', 1920, 1440)

            head_point = cv2.selectROI(
                'Select Head', img, fromCenter=False, showCrosshair=True)
            cv2.destroyAllWindows()
            head_x, head_y = head_point[0], head_point[1]
            # print("head_x", head_x)
            # print("head_y", head_y)
            # print("first_frame['x_head']", first_frame['x_head'])
            # print("first_frame['y_head']", first_frame['y_head'])
            # print("first_frame['x_tail']", first_frame['x_tail'])
            # print("first_frame['y_tail']", first_frame['y_tail'])
            # print(np.sqrt((first_frame['x_head'] - head_x)
            #       ** 2 + (first_frame['y_head'] - head_y)**2))
            # print(np.sqrt((first_frame['x_tail'] - head_x)
            #       ** 2 + (first_frame['y_tail'] - head_y)**2))

            current_head_distances = np.sqrt(
                (first_frame['x_head'] - head_x)**2 + (first_frame['y_head'] - head_y)**2)
            current_tail_distances = np.sqrt(
                (first_frame['x_tail'] - head_x)**2 + (first_frame['y_tail'] - head_y)**2)

            if current_head_distances > current_tail_distances:
                interval_data.iloc[0] = switch_head_tail(interval_data.iloc[0])
                intervals.at[interval.name, 'switch_status'] = 1
            else:
                intervals.at[interval.name, 'switch_status'] = 0

        elif mode == 'auto':
            if interval['switch_status'] == 1:
                # Switch the head and tail of the first frame
                interval_data.iloc[0] = switch_head_tail(interval_data.iloc[0])

        # print("interval_data", len(interval_data))
        for i in range(1, len(interval_data)):
            j = i - 1
            while j >= 0 and (interval_data.iloc[j]['worm_status'] == 'Coiling or Splitted'):
                j -= 1

            if j >= 0:
                prev_row = interval_data.iloc[j]
            else:
                # Fallback if no valid frame is found, should not happen in regular cases
                assert False, "No valid frame found"

            curr_row = interval_data.iloc[i]

            prev_head = np.array([prev_row['x_head'], prev_row['y_head']])
            curr_head = np.array([curr_row['x_head'], curr_row['y_head']])
            curr_tail = np.array([curr_row['x_tail'], curr_row['y_tail']])

            if np.linalg.norm(curr_head - prev_head) > np.linalg.norm(curr_tail - prev_head):
                interval_data.iloc[i] = switch_head_tail(curr_row)

        raw_data.update(interval_data)

    raw_data.to_csv('modified_raw_data.csv', index=False)
    intervals.to_csv('intervals.csv', index=False)


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
