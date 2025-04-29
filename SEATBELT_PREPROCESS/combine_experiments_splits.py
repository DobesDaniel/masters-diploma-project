import os
import shutil

from pathlib import Path

def merge_directories(source1, source2, target):
    os.makedirs(target, exist_ok=True)

    for source in [source1, source2]:
        for experiment in os.listdir(source):
            experiment_path = os.path.join(source, experiment)

            if not os.path.isdir(experiment_path):
                continue

            for subset in ['train', 'test']:        # could be also 'val'
                subset_path = os.path.join(experiment_path, subset)
                if not os.path.isdir(subset_path):
                    continue

                for class_name in os.listdir(subset_path):
                    class_path = os.path.join(subset_path, class_name)
                    if not os.path.isdir(class_path):
                        continue

                    target_class_path = os.path.join(target, experiment, subset, class_name)
                    os.makedirs(target_class_path, exist_ok=True)

                    for file in os.listdir(class_path):
                        if file.endswith('.npy'):               # expected file type...
                            source_file_path = os.path.join(class_path, file)
                            target_file_path = os.path.join(target_class_path, file)

                            if os.path.exists(target_file_path):
                                base, ext = os.path.splitext(file)
                                count = 1
                                while os.path.exists(target_file_path):
                                    new_filename = f"{base}_{count}{ext}"
                                    target_file_path = os.path.join(target_class_path, new_filename)
                                    count += 1

                            shutil.copy2(source_file_path, target_file_path)
                            print(f"Copied to {target_file_path}")


if __name__ == "__main__":

    # example
    source1 = '../resources/SB_experiments_split'
    source2 = '../resources/MY_SB_experiments_split'
    target = '../resources/Combined_SB_experiments_split'
    merge_directories(source1, source2, target)

    pass