import numpy as np
from tqdm import tqdm 

import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from torch.utils.tensorboard import SummaryWriter
import torch.optim as optim

from sklearn.base import BaseEstimator
from sklearn.metrics import r2_score, accuracy_score


# Add batch_norm
class TestNetwork(nn.Module, BaseEstimator):
    """docstring for CompositeNetwork"""
    def __init__(self, lr=3e-4):
        super(TestNetwork, self).__init__()
        self.lr = lr
        self.criterion = None
        self.type = None

        self.n_epoch = 100
        
        self.tqdm_disable = False
        self.writer = None

    def construct_network(self, data):
        x, y = data[:]
        with torch.no_grad():
            self.set_type(y)
            self.n_inputs = x.shape[1]
            self.n_channels = [200, 100, 50, 50, 50]

            layers = []
            for i, channels in enumerate(self.n_channels):
                in_channels = self.n_inputs if i == 0 else self.n_channels[i-1]
                out_channels = self.n_channels[i]
                layers += [nn.Linear(in_channels, out_channels), nn.ReLU()]
            layers += [nn.Linear(self.n_channels[-1], self.n_outputs)]
            self.network = nn.Sequential(*layers)

    def set_type(self, y):
            if y.dtype == torch.int64:
                self.type = "categorical"
                self.criterion = nn.CrossEntropyLoss()
                self.n_outputs = torch.unique(y).shape[0]
            elif y.dtype == torch.float32:
                self.type = "continuous"
                self.criterion = nn.MSELoss()
                self.n_outputs = 1
            else:
                raise

    def forward(self, x):
        return self.network(x)

    def fit(self, data):
        self.construct_network(data)
        train_data, val_data = data.train_test_split()
        train_loader = DataLoader(train_data, batch_size=32, shuffle=True)
        opt = optim.Adam(self.parameters(), lr=self.lr)

        for epoch in tqdm(range(self.n_epoch), disable=self.tqdm_disable):

            # TRAIN MODEL ON TRAIN DATA
            for b_x, b_y in train_loader:
                self.train()
                opt.zero_grad()
                # Forward pass, calculate loss, backward pass
                b_y_hat = self.forward(b_x)
                loss = self.loss(b_y_hat, b_y)
                loss.backward()
                opt.step()
            
            # EVALUATE MODEL ON VALIDATION DATA
            with torch.no_grad():
                self.eval()
                self.evaluate(train_data, val_data, epoch)
                
    def loss(self, y_hat, y):
        if self.type == "continuous":
            y = y.view((y.shape[0], 1))
        return self.criterion(y_hat, y)

    def score(self, y_hat, y):
        if self.type == "categorical":
            y_hat = np.argmax(y_hat, axis=1)
            score = accuracy_score(y, y_hat)
        elif self.type == "continuous":
            score = r2_score(y, y_hat)
        else:
            raise
        return score
    
    def evaluate(self, train_data, test_data, epoch):
        train_x, train_y = train_data[:]
        test_x, test_y = test_data[:]

        train_y_hat = self.forward(train_x)
        test_y_hat = self.forward(test_x)

        train_loss = self.loss(train_y_hat, train_y)
        test_loss = self.loss(test_y_hat, test_y)

        train_score = self.score(train_y_hat, train_y)
        test_score = self.score(test_y_hat, test_y)

        if self.writer:
            self.writer.add_scalars("Loss", 
                                   {"train" : train_loss, 
                                    "test" : test_loss},
                                    epoch)
            self.writer.add_scalars("Score", 
                                   {"train" : train_score, 
                                    "test" : test_score},
                                    epoch)
            if epoch == self.n_epoch-1:
                extractor = self.network[:-3]
                latent_test_x = extractor(test_x)
                self.writer.add_embedding(latent_test_x)
                self.writer.add_graph(self.network, input_to_model=test_x)
                self.writer.close()
