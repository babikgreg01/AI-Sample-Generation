import numpy as np
import os
import torch
import torchvision
import torchvision.transforms as transforms
from torch.utils.data import Dataset, DataLoader
from torch.utils.data import random_split
import torch.optim as optim
import torch.nn as nn
import matplotlib.pyplot as plt
from torch.utils.data import WeightedRandomSampler

from torchview import draw_graph


    
class NpyDataset(Dataset):
    def __init__(self, data_folder, transform=None):
        self.data_folder = data_folder
        self.classes = sorted(os.listdir(data_folder))
        self.class_to_idx = {cls: idx for idx, cls in enumerate(self.classes)}
        self.npy_files = self._get_npy_files()
        self.transform = transform

    def _get_npy_files(self):
        npy_files = []
        for cls in self.classes:
            class_path = os.path.join(self.data_folder, cls)
            if os.path.isdir(class_path):
                for f in os.listdir(class_path):
                    if f.endswith('.npy'):
                        npy_files.append((cls, f))
        return npy_files

    def __len__(self):
        return len(self.npy_files)

    def __getitem__(self, idx):
        cls, npy_file = self.npy_files[idx]
        npy_file_path = os.path.join(self.data_folder, cls, npy_file)
        sample = torch.from_numpy(np.load(npy_file_path))
        
    
        # label is index of class
        label = self.class_to_idx[cls]

        if self.transform:
            sample = self.transform(sample)

        # adding channel dimension for CNN input
        sample = torch.unsqueeze(sample, 0)

        return sample, label