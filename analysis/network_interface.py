import copy
from math import sqrt
import numpy as np
import pandas as pd
from tqdm import tqdm 
import matplotlib.pyplot as plt

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.nn.utils import weight_norm
import torch.optim as optim

from sklearn.model_selection import train_test_split

class Network(object):

    def __init__(self):
        super(Network, self).__init__()
        self.lr = None
        self.criterion = None
        self.network = None
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

        self.n_epoch = 150
        self.val_frac = 0.2
        self.patience = 150

        cols = ["Epoch", "Training error", "Validation error"]
        array = np.empty((self.n_epoch, len(cols)))
        array[:] = np.nan
        self.history = pd.DataFrame(array, columns=cols)
        self.tqdm_disable = True

        self.subject_wise_split = False
        self.target_string = None

    def learn(self, x, y):
        # Implement subject-wise train/val split
        if self.subject_wise_split:
            from utils import subject_wise_split
            print("HEEJ")
            x_train, x_val, y_train, y_val = subject_wise_split(x, y,
                                                          self.target_string,
                                                          test_size=self.val_frac)
            print("HEEJ")
        else:
            x_train, x_val, y_train, y_val = train_test_split(x, y, 
                                                          test_size=self.val_frac,
                                                          stratify=y)

        

        # Initialize training/validation loop
        val_counter = 0
        best_loss = torch.Tensor([np.inf])
        best_model = None
        opt = optim.Adam(self.parameters(), self.lr)

        # Training/validation loop
        for epoch in tqdm(range(self.n_epoch), disable=self.tqdm_disable):
            self.train()
            opt.zero_grad()
            y_hat_train = self.forward(x_train)
            train_loss = self.criterion(y_hat_train, y_train)
            train_loss.backward()
            opt.step()

            with torch.no_grad():
                self.eval()
                y_hat_val = self.forward(x_val)
                val_loss = self.criterion(y_hat_val, y_val)

            train_score = self.score(x_train, y_train)
            val_score = self.score(x_val, y_val)
            self.history.loc[epoch, "Epoch"] = epoch
            self.history.loc[epoch, "Training Error"] = train_score
            self.history.loc[epoch, "Validation Error"] = val_score

            if val_loss < best_loss:
                best_model = copy.deepcopy(self.state_dict())
                best_loss = val_loss
                val_counter = 0
            else:
                val_counter += 1
                if val_counter == self.patience:
                    self.load_state_dict(best_model)
                    self.history.dropna(how="all")
                    break
        
    def forward(self, x):
        return self.network(x)