import torch
from torch.utils.data import Dataset
import numpy as np


class CustomDataset(Dataset):
    def __init__(self, data):
        self.X = data.item()['X']  # Extracting the features
        self.Y = data.item()['Y']  # Extracting the labels

    def __len__(self):
        return len(self.X)

    def __getitem__(self, idx):
        # Here, you might need to do some processing depending on your data's format
        # For example, you might need to convert arrays to PyTorch tensors
        return (torch.tensor(self.X[idx], dtype=torch.float32),
                torch.tensor(self.Y[idx], dtype=torch.long))


class OverlappingDataset(Dataset):
    def __init__(self, npz_file):
        data = np.load(npz_file)
        self.X = data['features']  # Assuming the features are stored under key 'features'
        self.Y = data['labels']    # Assuming the labels are stored under key 'labels'

    def __len__(self):
        return len(self.X)

    def __getitem__(self, idx):
        # Converting arrays to PyTorch tensors
        return (torch.tensor(self.X[idx], dtype=torch.float32),
                torch.tensor(self.Y[idx], dtype=torch.float32))
