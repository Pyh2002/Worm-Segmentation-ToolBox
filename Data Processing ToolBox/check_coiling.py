import csv

def check_worm_status(file_path):
    with open(file_path, 'r') as file:
        reader = csv.DictReader(file)
        status_list = []
        for row in reader:
            status_list.append((int(row['frame_number']), row['worm_status']))

    for i in range(len(status_list) - 4):
        if all(status_list[j][1] == 'Coiling or Splitted' for j in range(i, i + 5)):
            print(status_list[i][0])

check_worm_status('modified_raw_data.csv')
