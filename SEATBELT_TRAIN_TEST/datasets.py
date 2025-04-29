import os
import numpy as np
import torch
from torch.utils.data import Dataset


class MyDataset(Dataset):
    def __init__(self, root_dir, transform=None):
        self.root_dir = root_dir
        self.transform = transform
        self.classes = sorted(os.listdir(root_dir))
        self.class_to_idx = {class_name: idx for idx, class_name in enumerate(self.classes)}
        self.data = []

        # load all the .npy files from each class folder
        for class_name in self.classes:
            class_folder = os.path.join(root_dir, class_name)
            for file_name in os.listdir(class_folder):
                if file_name.endswith('.npy'):
                    file_path = os.path.join(class_folder, file_name)
                    self.data.append((file_path, self.class_to_idx[class_name]))

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        file_path, label = self.data[idx]
        values = np.load(file_path)  # Load the numerical data from .npy file

        values = torch.tensor(values, dtype=torch.float32)

        if self.transform:
            values = self.transform(values)

        return values, label