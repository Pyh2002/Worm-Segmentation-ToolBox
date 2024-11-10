import pandas as pd


def group_images(input_csv='raw_data.csv', output_csv='intervals.csv'):
    df = pd.read_csv(input_csv, delimiter=',')

    intervals = []
    start_index = None
    current_status = None

    for index, row in df.iterrows():
        status = row['status']
        if status == 'Regular' or status == 'Multiple':
            if start_index is None:
                start_index = index
                current_status = status
            elif status != current_status:
                end_index = index - 1
                if end_index - start_index >= 50:
                    while df.iloc[start_index]['worm_status'] == 'Coiling or Splitted':
                        start_index += 1
                        if start_index >= index:
                            start_index = None
                            end_index = None
                            break
                    while df.iloc[end_index]['worm_status'] == 'Coiling or Splitted':
                        end_index -= 1
                        if end_index <= start_index:
                            start_index = None
                            end_index = None
                            break
                    if start_index is not None and end_index is not None:
                        start_frame = df.iloc[start_index]['frame_number']
                        end_frame = df.iloc[end_index]['frame_number']
                        intervals.append(
                            [start_frame, end_frame, end_index - start_index + 1, current_status, 1, 0])
                        if current_status == 'Multiple':
                            intervals.append(
                                [start_frame, end_frame, end_index - start_index + 1, current_status, 2, 0])
                start_index = index
                current_status = status
        else:
            if start_index is not None:
                end_index = index - 1
                if end_index - start_index >= 50:
                    while df.iloc[start_index]['worm_status'] == 'Coiling or Splitted':
                        start_index += 1
                        if start_index >= index:
                            start_index = None
                            end_index = None
                            break
                    while df.iloc[end_index]['worm_status'] == 'Coiling or Splitted':
                        end_index -= 1
                        if end_index <= start_index:
                            start_index = None
                            end_index = None
                            break
                    if start_index is not None and end_index is not None:
                        start_frame = df.iloc[start_index]['frame_number']
                        end_frame = df.iloc[end_index]['frame_number']
                        intervals.append(
                            [start_frame, end_frame, end_index - start_index + 1, current_status, 1, 0])
                        if current_status == 'Multiple':
                            intervals.append(
                                [start_frame, end_frame, end_index - start_index + 1, current_status, 2, 0])
            start_index = None
            end_index = None

    if start_index is not None:
        end_index = len(df) - 1
        if end_index - start_index >= 50:
            while df.iloc[start_index]['worm_status'] == 'Coiling or Splitted':
                start_index += 1
                if start_index >= len(df):
                    break
            while df.iloc[end_index]['worm_status'] == 'Coiling or Splitted':
                end_index -= 1
                if end_index <= start_index:
                    start_index = None
                    end_index = None
                    break
            if start_index is not None and end_index is not None:
                start_frame = df.iloc[start_index]['frame_number']
                end_frame = df.iloc[end_index]['frame_number']
                intervals.append(
                    [start_frame, end_frame, end_index - start_index + 1, current_status, 1, 0])
                if current_status == 'Multiple':
                    intervals.append(
                        [start_frame, end_frame, end_index - start_index + 1, current_status, 2, 0])

    intervals_df = pd.DataFrame(
        intervals, columns=['start_frame', 'end_frame', 'num_frames', 'status', 'worm_id', 'switch_status'])

    intervals_df.to_csv(output_csv, index=False)
