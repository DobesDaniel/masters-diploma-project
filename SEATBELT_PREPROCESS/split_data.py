"""
    This script is creating data splits for train and test

"""

import os
import shutil
import random
import logging

def setup_logging(output_dir):
    """Set up logging to log to both console and file."""
    log_file_path = os.path.join(output_dir, 'split_log.txt')
    logging.basicConfig(level=logging.INFO,
                        format='%(asctime)s - %(message)s',
                        handlers=[
                            logging.StreamHandler(),
                            logging.FileHandler(log_file_path)
                        ])

def split_experiment_data(experiment_dir, output_dir, split_ratio=0.8):
    """
    Splits data into train and test sets, preserving class directory structure.

    Parameters:
        experiment_dir (str): Path to the experiment directory.
        output_dir (str): Path to the output directory.
        split_ratio (float): Ratio of files to include in the train set (default: 0.8).
    """
    # reproducibility - this can be removed or changed if trying for different split
    random.seed(42)

    train_dir = os.path.join(output_dir, "train")
    test_dir = os.path.join(output_dir, "test")
    os.makedirs(train_dir, exist_ok=True)
    os.makedirs(test_dir, exist_ok=True)

    for class_name in os.listdir(experiment_dir):
        class_dir = os.path.join(experiment_dir, class_name)
        if not os.path.isdir(class_dir):
            continue

        train_class_dir = os.path.join(train_dir, class_name)
        test_class_dir = os.path.join(test_dir, class_name)
        os.makedirs(train_class_dir, exist_ok=True)
        os.makedirs(test_class_dir, exist_ok=True)

        # data to split - .npy
        files = [f for f in os.listdir(class_dir) if f.endswith('.npy')]
        random.shuffle(files)

        split_index = int(len(files) * split_ratio)

        train_files = files[:split_index]
        test_files = files[split_index:]

        for f in train_files:
            src = os.path.join(class_dir, f)
            dst = os.path.join(train_class_dir, f)
            shutil.copy(src, dst)

        for f in test_files:
            src = os.path.join(class_dir, f)
            dst = os.path.join(test_class_dir, f)
            shutil.copy(src, dst)

        logging.info(f"Class '{class_name}': {len(train_files)} train, {len(test_files)} test files")

    logging.info(f"Data split completed. Train and test sets are saved in '{output_dir}'.")



if __name__ == "__main__":

    setup_logging("./")

    input_base_dir = f"../resources/MY_SB_experiments/"
    output_base_dir = f"../resources/MY_SB_experiments_split/"
    split_ratio = 0.8

    for experiment_name in os.listdir(input_base_dir):
        experiment_dir = os.path.join(input_base_dir, experiment_name)
        output_dir = os.path.join(output_base_dir, experiment_name)

        if os.path.isdir(experiment_dir):
            split_experiment_data(experiment_dir, output_dir, split_ratio)

    # experiment_name = "size_3_featuresA"    #
    # experiment_dir = f"../resources/SB_experiments/{experiment_name}"
    # output_dir = f"../resources/SB_experiments_split/{experiment_name}"
    # split_ratio = 0.8
    # split_experiment_data(experiment_dir, output_dir, split_ratio)

    pass