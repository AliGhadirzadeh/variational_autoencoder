import numpy as np
from tqdm import tqdm 

import torch
import torch.nn as nn
from torch.nn.utils import weight_norm
from torch.utils.data import DataLoader
from torch.utils.tensorboard import SummaryWriter
import torch.optim as optim

from sklearn.base import BaseEstimator
from sklearn.metrics import r2_score


class Probe_Y(nn.Module):

    def __init__(self):
        super(Probe_Y, self).__init__()
        self.lr = 3e-4
        self.epoch = 1000
        n_input = 12
        n_output = 1
        self.network = nn.Sequential(nn.Linear(n_input, 500), nn.ReLU(),
                                     nn.Linear(500, 500), nn.ReLU(),
                                     nn.Linear(500, 500), nn.ReLU(),
                                     nn.Linear(500, 500), nn.ReLU(),
                                     nn.Linear(500, 500), nn.ReLU(),
                                     nn.Linear(500, n_output))
        self.criterion = nn.MSELoss()

        self.tqdm_disable = False
        self.writer = None
        self.print = True

    def forward(self, x):
        return self.network(x)

    def loss(self, y_hat, y):
        y = y.view((y.shape[0], 1))
        return self.criterion(y_hat, y)

    def score(self, y_hat, y):
        return r2_score(y, y_hat)

    def fit(self, data):
        train_data, val_data = data.train_test_split()
        train_loader = DataLoader(train_data, batch_size=32, shuffle=True)
        opt = optim.Adam(self.parameters(), lr=self.lr)

        for epoch in tqdm(range(self.epoch), disable=self.tqdm_disable):
            for b_z, b_y in train_loader:
                self.train()
                opt.zero_grad()
                b_y_hat = self.network(b_z)
                loss = self.loss(b_y_hat, b_y)
                loss.backward()
                opt.step()

            with torch.no_grad():
                self.eval()
                self.evaluate_probe(train_data, val_data, epoch)

    def evaluate_probe(self, train_data, test_data, epoch):
        train_z, train_y = train_data[:]
        test_z, test_y = test_data[:]

        train_y_hat = self.network(train_z)

        test_y_hat = self.network(test_z)

        train_loss = self.loss(train_y_hat, train_y)
        test_loss = self.loss(test_y_hat, test_y)

        train_score = self.score(train_y_hat, train_y)
        test_score = self.score(test_y_hat, test_y)

        if self.print:
            print("Train score (probe): " + "{:+.3f}".format(train_score))
            print("Test score (probe):  " + "{:+.3f}".format(test_score))

        if self.writer:
            self.writer.add_scalars("Probe Loss", {"train_y" : train_loss,
                                             "test_y" : test_loss}, 
                                             epoch)
            self.writer.add_scalars("Probe Score", {"train_y" : train_score,
                                              "test_y" : test_score},
                                              epoch)