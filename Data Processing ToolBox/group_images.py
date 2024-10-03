import pandas as pd


def group_images(input_csv='raw_data.csv', output_csv='intervals.csv'):
    df = pd.read_csv(input_csv, delimiter=',')

    intervals = []
    start_index = None
    current_status = None

    for index, row in df.iterrows():
        status = row['status']

        if status in ['Regular', 'Multiple']:
            if start_index is None:
                start_index = row['frame_number']
                current_status = status
            elif current_status != status:
                end_index = df.iloc[index - 1]['frame_number']
                num_frames = end_index - start_index + 1
                if num_frames >= 50:
                    intervals.append(
                        [start_index + 5, end_index - 5, num_frames - 10, current_status])
                start_index = row['frame_number']
                current_status = status
        else:
            if start_index is not None:
                end_index = df.iloc[index - 1]['frame_number']
                num_frames = end_index - start_index + 1
                if num_frames >= 50:
                    intervals.append(
                        [start_index + 5, end_index - 5, num_frames - 10, current_status])
                start_index = None

    if start_index is not None:
        end_index = df.iloc[-1]['frame_number']
        num_frames = end_index - start_index + 1
        if num_frames >= 10:
            intervals.append([start_index + 5, end_index - 5,
                              num_frames - 10, current_status])

    intervals_df = pd.DataFrame(
        intervals, columns=['start_frame', 'end_frame', 'num_frames', 'status'])

    intervals_df.to_csv(output_csv, index=False)
