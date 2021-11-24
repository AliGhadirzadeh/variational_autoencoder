import numpy as np
import torch
from torch.utils.data import Dataset


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


class SubjectDataset(Dataset):

    def __init__(self, x, y, ids, binned=False, n_levels=3):
        self.x = x
        self.y = self.bin(y, n_levels=n_levels) if binned else y
        self.ids = ids
        self.subject_wise = False
        self.test_size = 0.2

    def __len__(self):
        return self.x.shape[0]

    def __getitem__(self, index):
        x_sample = self.x[index]
        y_sample = self.y[index]
        id_sample = self.ids[index]
        return x_sample, y_sample, id_sample

    def train_test_split(self):
        if self.subject_wise:
            data_tuple = self.subject_wise_split(self.x, self.y, self.ids,
                                                 stratify=True)
        else:
            from sklearn.model_selection import train_test_split
            data_tuple = train_test_split(self.x, self.y, self.ids,
                                     test_size=self.test_size,
                                     stratify=self.bin(self.y))
        x_train, y_train, ids_train = data_tuple[0::2]
        x_test, y_test, ids_test = data_tuple[1::2]
        train_data = SubjectDataset(x_train, y_train, ids_train)
        test_data = SubjectDataset(x_test, y_test, ids_test)
        return train_data, test_data

    def bin(self, y, n_levels=3):
        binned_vals = torch.zeros(y.shape[0], dtype=torch.int64)
        unique_vals = torch.unique(y)
        chunks = unique_vals.chunk(n_levels)
        for i, chunk in enumerate(chunks):
            binned_vals[self.isin(y, chunk)] = i
        return binned_vals

    def subject_wise_split(self, x, y, ids, stratify=True):
        unique_ids = torch.unique(ids)
        unique_vals = torch.zeros(unique_ids.shape[0])
        for i in unique_ids:
            val = self.y[torch.eq(self.ids, i)][0]
            unique_vals[i] = val
        binned_vals = self.bin(unique_vals)
        from sklearn.model_selection import train_test_split
        if stratify:
            vals_train, vals_test = train_test_split(unique_ids, test_size=self.test_size,
                                                     stratify=binned_vals)
        else:
            vals_train, vals_test = train_test_split(unique_ids, test_size=self.test_size)

        idx_train = self.isin(self.ids, vals_train)
        idx_test = self.isin(self.ids, vals_test)

        x_train = self.x[idx_train]
        y_train = self.y[idx_train]
        ids_train = self.ids[idx_train]

        x_test = self.x[idx_test]
        y_test = self.y[idx_test]
        ids_test = self.ids[idx_test]

        return (x_train, x_test, y_train, y_test, ids_train, ids_test)

    def isin(self, ar1, ar2):
        return (ar1[..., None] == ar2).any(-1)
