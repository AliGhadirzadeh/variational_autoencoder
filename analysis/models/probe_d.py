import numpy as np
from tqdm import tqdm 

import torch
import torch.nn as nn
from torch.nn.utils import weight_norm
from torch.utils.data import DataLoader
from torch.utils.tensorboard import SummaryWriter
import torch.optim as optim

from sklearn.base import BaseEstimator
from sklearn.metrics import accuracy_score


class Probe_D(nn.Module):

    def __init__(self):
        super(Probe_D, self).__init__()
        self.lr = 3e-4
        self.epoch = 1000
        n_input = 12
        n_output = 42
        self.network = nn.Sequential(nn.Linear(n_input, 500), nn.ReLU(),
                                     nn.Linear(500, 500), nn.ReLU(),
                                     nn.Linear(500, 500), nn.ReLU(),
                                     nn.Linear(500, 500), nn.ReLU(),
                                     nn.Linear(500, 500), nn.ReLU(),
                                     nn.Linear(500, n_output))
        self.criterion = nn.CrossEntropyLoss()
        self.tqdm_disable = False
        self.writer = None
        self.print = True

    def forward(self, x):
        return self.network(x)

    def loss(self, d_hat, d):
        return self.criterion(d_hat, d)

    def score(self, d_hat, d):
        d_hat = np.argmax(d_hat, axis=1)
        return accuracy_score(d, d_hat)

    def fit(self, data):
        train_data, val_data = data.train_test_split()
        train_loader = DataLoader(train_data, batch_size=32, shuffle=True)
        opt = optim.Adam(self.parameters(), lr=self.lr)

        for epoch in tqdm(range(self.epoch), disable=self.tqdm_disable):
            for b_z, b_d in train_loader:
                self.train()
                opt.zero_grad()
                b_d_hat = self.network(b_z)
                loss = self.loss(b_d_hat, b_d)
                loss.backward()
                opt.step()

            with torch.no_grad():
                self.eval()
                self.evaluate_probe(train_data, val_data, epoch)

    def evaluate_probe(self, train_data, test_data, epoch):
        train_z, train_d = train_data[:]
        test_z, test_d = test_data[:]

        train_d_hat = self.network(train_z)

        test_d_hat = self.network(test_z)

        train_loss = self.loss(train_d_hat, train_d)
        test_loss = self.loss(test_d_hat, test_d)

        train_score = self.score(train_d_hat, train_d)
        test_score = self.score(test_d_hat, test_d)

        if self.print:
            print("Train score (probe): " + "{:+.3f}".format(train_score))
            print("Test score (probe):  " + "{:+.3f}".format(test_score))

        if self.writer:
            self.writer.add_scalars("Probe Loss", {"train_d" : train_loss,
                                             "test_d" : test_loss}, 
                                             epoch)
            self.writer.add_scalars("Probe Score", {"train_d" : train_score,
                                              "test_d" : test_score},
                                              epoch)