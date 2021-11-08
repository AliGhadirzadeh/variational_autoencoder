import numpy as np
import torch
from torch.utils.data import Dataset, DataLoader


class CtrlDataset(Dataset):

    def __init__(self, x, y, c, binned=False):
        self.x = x
        self.y = self.bin(y) if binned else y
        self.c = c

    def __len__(self):
        return self.x.shape[0]

    def __getitem__(self, index):
        x_sample = self.x[index]
        y_sample = self.y[index]
        c_sample = self.c[index]
        return x_sample, y_sample, c_sample 

    def train_test_split(self, test_size=0.2):
        from sklearn.model_selection import train_test_split
        data_list = train_test_split(self.x, self.y, self.c, 
                                     test_size=test_size,
                                     stratify=self.bin(self.y))
        x_train, y_train, c_train = data_list[0::2]
        x_test, y_test, c_test = data_list[1::2]
        train_data = CtrlDataset(x_train, y_train, c_train)
        test_data = CtrlDataset(x_test, y_test, c_test)
        return train_data, test_data

    def bin(self, y, n_levels=3):
        binned_vals = torch.zeros(len(self), dtype=torch.int64)
        unique_vals = torch.unique(y)
        chunks = unique_vals.chunk(n_levels)
        for i, chunk in enumerate(chunks):
            binned_vals[self.isin(y, chunk)] = i
        return binned_vals

    def isin(self, ar1, ar2):
        return (ar1[..., None] == ar2).any(-1)            


class Dataset(Dataset):

    def __init__(self, x, y, binned=False):
        self.x = x
        self.y = self.bin(y) if binned else y

    def __len__(self):
        return self.x.shape[0]

    def __getitem__(self, index):
        x_sample = self.x[index]
        y_sample = self.y[index]
        return x_sample, y_sample 

    def train_test_split(self, test_size=0.2):
        from sklearn.model_selection import train_test_split
        data_list = train_test_split(self.x, self.y, 
                                     test_size=test_size,
                                     stratify=self.bin(self.y))
        x_train, y_train = data_list[0::2]
        x_test, y_test = data_list[1::2]
        train_data = Dataset(x_train, y_train)
        test_data = Dataset(x_test, y_test)
        return train_data, test_data

    def bin(self, y, n_levels=3):
        binned_vals = torch.zeros(len(self), dtype=torch.int64)
        unique_vals = torch.unique(y)
        chunks = unique_vals.chunk(n_levels)
        for i, chunk in enumerate(chunks):
            binned_vals[self.isin(y, chunk)] = i
        return binned_vals

    def isin(self, ar1, ar2):
        return (ar1[..., None] == ar2).any(-1) 