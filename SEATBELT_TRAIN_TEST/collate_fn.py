import torch
from torch.nn.utils.rnn import pad_sequence

def collate_fn(batch):
    keypoints, labels = zip(*batch)
    keypoints = pad_sequence(keypoints, batch_first=True, padding_value=0)  # Pad sequences to same length
    labels = torch.tensor(labels)
    return keypoints, labels
