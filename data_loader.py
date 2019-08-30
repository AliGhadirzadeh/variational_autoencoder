from random import shuffle

import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader
from torch.utils.data.sampler import SubsetRandomSampler
import models
import vae

import matplotlib.pyplot as plt
import numpy as np

import argparse


class EEGDataset(Dataset):
    def __init__(self, data_filename, shuffle=True, device="cpu"):
        self.device = device
        numpy_data = np.load(data_filename)
        if shuffle:
        	np.random.shuffle(numpy_data)
        self.data = torch.from_numpy(numpy_data).float().to(self.device)
        self.num_samples = self.data.shape[0]
    
    def __len__(self):
        return self.num_samples

    def __getitem__(self, idx):
        return self.data[idx]

file_path = "/home/sebgho/eeg_project/raw_data/data/snippets/snippets.npy"
data_set = EEGDataset(file_path)

# Create samplers for training data and test data
test_fraction = 0.2
indices = list(range(len(data_set)))
split = int(np.floor(test_fraction * len(data_set)))
train_indices, test_indices = indices[split:], indices[:split]
train_sampler = SubsetRandomSampler(train_indices)
test_sampler = SubsetRandomSampler(test_indices)

# Create data loaders for test and training data
train_loader = DataLoader(data_set, batch_size=100,
						  sampler=train_sampler, num_workers=4)
test_loader = DataLoader(data_set, batch_size=100,
						 sampler=test_sampler, num_workers=4)





