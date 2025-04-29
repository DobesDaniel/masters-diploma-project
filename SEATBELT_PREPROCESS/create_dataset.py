"""

    This will create the datasets from CSV files containing landmarks for each frame of an action
    - action frame window = 120 frames (2 seconds) - each frame will have row for each video in file

    - its basically creates datasets with data create based on what is wanted
        - sequence length, features type, offset between frames...

    CSV files format (first line is header):
        {
                'frame': frame_idx,
                'nose_x': landmarks[mp_pose.PoseLandmark.NOSE].x,
                'nose_y': landmarks[mp_pose.PoseLandmark.NOSE].y,
                'left_shoulder_x': landmarks[mp_pose.PoseLandmark.LEFT_SHOULDER].x,
                'left_shoulder_y': landmarks[mp_pose.PoseLandmark.LEFT_SHOULDER].y,
                'right_shoulder_x': landmarks[mp_pose.PoseLandmark.RIGHT_SHOULDER].x,
                'right_shoulder_y': landmarks[mp_pose.PoseLandmark.RIGHT_SHOULDER].y,
                'left_elbow_x': landmarks[mp_pose.PoseLandmark.LEFT_ELBOW].x,
                'left_elbow_y': landmarks[mp_pose.PoseLandmark.LEFT_ELBOW].y,
                'right_elbow_x': landmarks[mp_pose.PoseLandmark.RIGHT_ELBOW].x,
                'right_elbow_y': landmarks[mp_pose.PoseLandmark.RIGHT_ELBOW].y,
                'left_wrist_x': landmarks[mp_pose.PoseLandmark.LEFT_WRIST].x,
                'left_wrist_y': landmarks[mp_pose.PoseLandmark.LEFT_WRIST].y,
                'right_wrist_x': landmarks[mp_pose.PoseLandmark.RIGHT_WRIST].x,
                'right_wrist_y': landmarks[mp_pose.PoseLandmark.RIGHT_WRIST].y
        }


    Input:
        ../resources/SB_cuts_landmarks/{class name}

    Output:
        ../resources/SB_experiments/size_{3-6}_features{A-E}/{class name}


"""

import os
import numpy as np
import csv
import time
from calculate_features import calculate_features


def create_experiment_data(lstm_block_count, frame_offset, feature_type, input_directory, output_directory):
    """
    Creates an experiment dataset by processing frames from CSV files containing landmarks.

    Args:
        lstm_block_count (int): Number of LSTM blocks of data there will (must be between 3 and 6).
        frame_offset (int): Frame offset (number of frames to skip).
        feature_type (str): The type of feature to extract ('A', 'B', 'C', 'D', or 'E').
        input_directory (str): The directory where input CSV files are stored.
        output_directory (str): The directory where output `.npy` files will be saved.

    Returns:
        None
    """

    if lstm_block_count < 3 or lstm_block_count > 6:
        raise ValueError('Features count must be between 3 and 6')

    if feature_type not in "ABCDE":
        raise ValueError("Invalid feature type")

    for class_name in os.listdir(input_directory):
        class_folder = os.path.join(input_directory, class_name)

        if os.path.isdir(class_folder):
            output_class_folder = os.path.join(output_directory,
                                               f"size_{lstm_block_count}_features{feature_type}", class_name)
            os.makedirs(output_class_folder, exist_ok=True)

            for csv_filename in os.listdir(class_folder):
                if csv_filename.endswith('.csv'):  # only .csv files
                    input_file = os.path.join(class_folder, csv_filename)

                    with open(input_file, mode='r') as csvfile:
                        reader = csv.DictReader(csvfile)
                        rows = list(reader)

                    # 1) Calculate feature values for each line (frame)
                    all_feature_values = []

                    for row in rows:
                        params = [
                            float(row['nose_x']), float(row['nose_y']),
                            float(row['left_shoulder_x']), float(row['left_shoulder_y']),
                            float(row['right_shoulder_x']), float(row['right_shoulder_y']),
                            float(row['left_elbow_x']), float(row['left_elbow_y']),
                            float(row['right_elbow_x']), float(row['right_elbow_y']),
                            float(row['left_wrist_x']), float(row['left_wrist_y']),
                            float(row['right_wrist_x']), float(row['right_wrist_y'])
                        ]

                        all_feature_values.append(calculate_features(feature_type, params))

                    # Step 2: Create lists for each feature_values and append following elements with offset in that list
                    for i in range(len(all_feature_values)):

                        block_array = []
                        for j in range(i, len(all_feature_values), frame_offset):
                            if len(block_array) < lstm_block_count:
                                block_array.append(all_feature_values[j])

                        if len(block_array) == lstm_block_count:
                            output_array = np.array(block_array)

                            output_filename = os.path.splitext(csv_filename)[0] + f"_block_{i}.npy"
                            output_file = os.path.join(output_class_folder, output_filename)

                            np.save(output_file, output_array)
                            # print(f"Experiment data saved to {output_file}")

def create_all_experiment_data():
    input_dir = '../resources/MY_SB_cuts_landmarks'
    output_dir = '../resources/MY_SB_experiments'


    import logging

    log_file_name = os.path.join(output_dir, 'log.txt')
    handlers = []
    console_handler = logging.StreamHandler()
    console_handler.setLevel(logging.DEBUG)
    handlers.append(console_handler)
    file_handler = logging.FileHandler(log_file_name, mode="a")
    file_handler.setLevel(logging.DEBUG)
    handlers.append(file_handler)
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s - %(levelname)s - %(message)s",
        handlers=handlers
    )
    logging.info(f"Logging initialized. Outputting to: {log_file_name}")


    # experiments_param = [(3,40),(4,30),(5,24),(6,20)]     # 60 fps
    experiments_param = [(3,20),(4,15),(5,12),(6,10)]       # 30 fps
    features_types = ['A', 'B', 'C', 'D', 'E']

    experiment_number = 0
    for lstm_block_count,frame_offset in experiments_param:
        for features_type in features_types:

            logging.info(f"Experiment: {experiment_number}")
            logging.info(f"Block count: {lstm_block_count}")
            logging.info(f"Frame offset: {frame_offset}")
            logging.info(f"Features calculation type: {features_type}")

            start_time = time.time()

            create_experiment_data(lstm_block_count=lstm_block_count,
                                   frame_offset=frame_offset,
                                   feature_type=features_type,
                                   input_directory=input_dir,
                                   output_directory=output_dir)

            # log the time taken for this experiment
            elapsed_time = time.time() - start_time
            logging.info(f"Experiment {experiment_number} completed in {elapsed_time:.2f} seconds.")
            logging.info(f"Experiment {experiment_number} completed in {elapsed_time / 60:.2f} minutes.")

            experiment_number += 1

# just a single experiment variation...
def main():
    create_experiment_data(lstm_block_count=3, frame_offset=40, feature_type='C',
                           input_directory='../resources/SB_cuts_landmarks',
                           output_directory='../resources/SB_experiments')


if __name__ == "__main__":
    # main()
    create_all_experiment_data()
    pass