import numpy as np
import pandas as pd
from tqdm import tqdm 

import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader
from torch.utils.tensorboard import SummaryWriter
import torch.optim as optim

from sklearn.base import BaseEstimator
from sklearn.metrics import r2_score, accuracy_score


class PredictorNetwork(nn.Module):

    def __init__(self, **kwargs):
        super(PredictorNetwork, self).__init__()
        self.network = None
        self.type = None

    def construct_network(self, x, y):
        self.set_type(y)
        num_inputs = x.shape[1]
        num_channels = [200, 100, 50]

        if self.type == "categorical":
            self.criterion = nn.CrossEntropyLoss()
            num_outputs = torch.unique(y).shape[0]
        elif self.type == "continuous":
            self.criterion = nn.MSELoss()
            num_outputs = 1
        else:
            raise

        layers = []
        for i, channels in enumerate(num_channels):
            in_channels = num_inputs if i == 0 else num_channels[i-1]
            out_channels = num_channels[i]
            layers += [nn.Linear(in_channels, out_channels), nn.ReLU()]
        layers += [nn.Linear(num_channels[-1], num_outputs)]
        self.network = nn.Sequential(*layers)

    def set_type(self, y):
        if y.dtype == torch.int64:
            self.type = "categorical"
        elif y.dtype == torch.float32:
            self.type = "continuous"
        else:
            raise

    def forward(self, x):
        return self.network(x)

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


class ExtractorNetwork(nn.Module):
    """docstring for Extractor_Network"""
    def __init__(self):
        super(ExtractorNetwork, self).__init__()
        self.network = None
        self.criterion = nn.MSELoss()
        self.conv = None

    def construct_network(self, x):
        self.conv = True if len(x.shape)==3 else False
        if self.conv:
            num_inputs = x.shape[1]
            conv_channels = [60, 40, 20]
            conv_kers = [10, 5, 2]
            layers = []
            for i in range(len(conv_channels)):
                in_channels = num_inputs if i == 0 else conv_channels[i-1]
                out_channels = conv_channels[i]
                ker_size = conv_kers[i]
                conv = nn.Conv1d(in_channels, out_channels, ker_size)
                layers += [conv, nn.ReLU(), nn.MaxPool1d(2)]
            layers += [nn.Flatten()]
            self.network = nn.Sequential(*layers)
        else:
            num_inputs = x.shape[1]
            num_channels = [100, 100, 100]
            num_outputs = 100
            layers = []
            for i in range(len(num_channels)):
                in_channels = num_inputs if i == 0 else num_channels[i-1]
                out_channels = num_channels[i]
                layers += [nn.Linear(in_channels, out_channels), nn.ReLU()]
            layers += [nn.Linear(num_channels[-1], num_outputs)]
            self.network = nn.Sequential(*layers)

    def forward(self, x):
        return self.network(x)


class ControlledNetwork(nn.Module, BaseEstimator):
    """docstring for CompositeNetwork"""
    def __init__(self, lr=3e-4, writer=None):
        super(ControlledNetwork, self).__init__()
        self.lr = lr

        self.extractor = ExtractorNetwork()
        self.dep_predictor = PredictorNetwork()
        self.ctrl_predictor = PredictorNetwork()

        self.n_epoch = 100
        
        self.tqdm_disable = False
        self.writer = writer

    def construct_network(self, data):
        with torch.no_grad():
            x, y, c = data[:]
            self.extractor.construct_network(x)
            x_latent = self.extractor(x)
            self.dep_predictor.construct_network(x_latent, y)
            self.ctrl_predictor.construct_network(x_latent, c)
            #if self.writer:
                #self.writer.add_graph(self)
            self.initialized = True

    def forward(self, x):
        latent_x = self.extractor(x)
        y = self.dep_predictor(latent_x)
        c = self.ctrl_predictor(latent_x)
        return y, c

    def fit(self, data):
        self.construct_network(data)
        train_data, val_data = data.train_test_split()
        train_loader = DataLoader(train_data, batch_size=32, shuffle=True)

        # One optimizer per network
        extractor_opt = optim.Adam(self.parameters(), lr=self.lr)
        dep_predictor_opt = optim.Adam(self.parameters(), lr=self.lr)
        ctrl_predictor_opt = optim.Adam(self.parameters(), lr=self.lr)

        for epoch in tqdm(range(self.n_epoch), disable=self.tqdm_disable):

            # TRAIN MODEL ON TRAIN DATA
            for b_x, b_y, b_c in train_loader:
                self.train()
                # ---- TRAIN EXTRACTOR ----
                # Activate extractor, de-activate rest, zero_grad
                self.deactivate(self.dep_predictor)
                self.deactivate(self.ctrl_predictor)
                self.activate(self.extractor)
                extractor_opt.zero_grad()
                # Forward pass, calculate loss, backward pass
                b_y_hat, b_c_hat = self.forward(b_x)
                extractor_loss = self.extractor_loss(b_y_hat, b_y, b_c_hat, b_c)
                extractor_loss.backward()
                extractor_opt.step()

                # ---- TRAIN DEP_PREDICTOR ----
                # Activate dep_predictor, de-activate rest, zero_grad
                self.deactivate(self.extractor)
                self.deactivate(self.ctrl_predictor)
                self.activate(self.dep_predictor)
                dep_predictor_opt.zero_grad()
                # Forward pass, calculate loss, backward pass
                b_y_hat, b_c_hat = self.forward(b_x)
                dep_predictor_loss = self.dep_predictor.loss(b_y_hat, b_y)
                dep_predictor_loss.backward()
                dep_predictor_opt.step()

                # ---- TRAIN CTRL_PREDICTORS ----
                # Activate each ctrl_predictor, de-activate rest, zero_grad
                self.deactivate(self.extractor)
                self.deactivate(self.dep_predictor)
                self.activate(self.ctrl_predictor)
                # Forward pass, calculate loss, backward pass
                ctrl_predictor_opt.zero_grad()
                b_y_hat, b_c_hat = self.forward(b_x)
                ctrl_predictor_loss = self.ctrl_predictor.loss(b_c_hat, b_c)
                ctrl_predictor_loss.backward()
                ctrl_predictor_opt.step()
            
            # EVALUATE MODEL ON VALIDATION DATA
            with torch.no_grad():
                self.eval()
                self.evaluate(train_data, val_data, epoch)

    def activate(self, network):
        for param in network.parameters():
            param.requires_grad = True

    def deactivate(self, network):
        for param in network.parameters():
            param.requires_grad = False
                
    def extractor_loss(self, y_hat, y, c_hat, c):
        dep_loss = self.dep_predictor.loss(y_hat, y)
        ctrl_loss = self.ctrl_predictor.loss(c_hat, c)

        # Extractor trained to minimize composite loss
        extractor_loss = dep_loss + 1 * ctrl_loss
        return extractor_loss

    def score(self, data):
        with torch.no_grad():
            x, y, c = data[:]
            y_hat, c_hat = self.forward(x)
            score = self.ctrl_predictor.score(c_hat, c)
        return score.mean()
    
    def evaluate(self, train_data, test_data, epoch):
        train_x, train_y, train_c = train_data[:]
        test_x, test_y, test_c = test_data[:]
        
        train_y_hat, train_c_hat = self.forward(train_x)
        test_y_hat, test_c_hat = self.forward(test_x)

        extractor_loss = self.extractor_loss(test_y_hat, test_y, test_c_hat, test_c)
        dep_loss = self.dep_predictor.loss(test_y_hat, test_y)
        ctrl_loss = self.ctrl_predictor.loss(test_c_hat, test_c)

        train_dep_score = self.dep_predictor.score(train_y_hat, train_y)
        train_ctrl_score = self.ctrl_predictor.score(train_c_hat, train_c)
        test_dep_score = self.dep_predictor.score(test_y_hat, test_y)
        test_ctrl_score = self.ctrl_predictor.score(test_c_hat, test_c)

        if self.writer:
            self.writer.add_scalars("Loss", {"extractor" : extractor_loss,
                                             "dep_predictor" : dep_loss,
                                             "ctrl_predictor" : ctrl_loss}, 
                                             epoch)
            self.writer.add_scalars("Score/dep_predictor", {"train" : train_dep_score, 
                                                            "test" : test_dep_score},
                                                            epoch)
            self.writer.add_scalars("Score/ctrl_predictor", {"train" : train_ctrl_score,
                                                             "test" : test_ctrl_score},
                                                              epoch)


class Dataset(Dataset):

    def __init__(self, x, y, c):
        self.x = x
        self.y = y
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
                                     test_size=test_size)
        x_train, y_train, c_train = data_list[0::2]
        x_test, y_test, c_test = data_list[1::2]
        train_data = Dataset(x_train, y_train, c_train)
        test_data = Dataset(x_test, y_test, c_test)
        return train_data, test_data