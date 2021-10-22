import copy
from math import sqrt
import numpy as np
import pandas as pd
from tqdm import tqdm 
import matplotlib.pyplot as plt

from sklearn.metrics import mean_squared_error, log_loss, r2_score, accuracy_score
from sklearn.base import BaseEstimator

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.nn.utils import weight_norm
import torch.optim as optim

from network_interface import Network


""" 
This file contains model classes that inherit from the sklearn.base BaseEstimator and
the Network super classes.

sklearn.base BaseEstimator is the sklearn Estimator interface, which is required for
compatibility with sklearn and skopt optimization algorithms. The Estimator interface
requires an implementation of fit(x, y), predict(x) and score(x, y). The Estimator
interface expects numpy.ndarray inputs in the format (N x dim_1 x dim_2 x ... x dim_n),
where N is the number of samples and dim_i is the i:th feature dimenson size.

Network is a neural network super class which provides implementations of neural
network-specific functionality, such as train(x, y), forward(x) and a plotting function.
The methods of the Network class expects pytorch.Tensor inputs.

To create a model that inherits from the above classes the following is required
    - Setting the lr and criterion attributes from the Network interface.
    - A construct_network(x, y) method, which constructs a neural network and sets the 
      network attribute from the Network interface.
    - An implementation of the Estimator interface fit(x, y) and predict(x) methods 
      using the train(x, y) and forward(x) methods implemented in the Network super class.
    - An implementation of the Estimator interface score(x, y) method.
"""

class FFNN_reg(BaseEstimator, Network, nn.Module):

    def __init__(self, fc_layer_1=200, fc_layer_2=200, fc_layer_3=100,
                 fc_layer_4=50, lr=3e-3, **kwargs):
        super(FFNN_reg, self).__init__()
        self.fc_layer_1 = fc_layer_1
        self.fc_layer_2 = fc_layer_2
        self.fc_layer_3 = fc_layer_3
        self.fc_layer_4 = fc_layer_4
        self.lr = lr
        self.criterion = nn.MSELoss()

    def construct_network(self, x, y):
        num_inputs = x.shape[1]
        num_outputs = 1
        fc_channels = []
        layers = []
        for key in self.__dict__:
            if "layer" in key:
                fc_channels += [self.__dict__[key]]
        for i in range(len(fc_channels)):
            in_channels = num_inputs if i == 0 else fc_channels[i-1]
            out_channels = fc_channels[i]
            layers += [nn.Linear(in_channels, out_channels), nn.ReLU()]
        layers += [nn.Linear(fc_channels[-1], num_outputs)]
        self.network = nn.Sequential(*layers)
        self.network.to(self.device)

    def fit(self, x, y):
        x = torch.Tensor(x).to(self.device)
        y = torch.Tensor(np.expand_dims(y, axis=1)).to(self.device)
        self.construct_network(x, y)
        self.learn(x, y)

    def predict(self, x):
        x = torch.Tensor(x)
        y_hat = self.forward(x).detach().numpy()
        return y_hat

    def score(self, x, y):
        x = torch.Tensor(x)
        y_hat = self.forward(x).detach().numpy()
        score = r2_score(y, y_hat)
        #score = - mean_squared_error(y, y_hat)
        return score

class FFNN_clf(BaseEstimator, Network, nn.Module):

    def __init__(self, fc_layer_1=100, fc_layer_2=100, fc_layer_3=100,
                 fc_layer_4=100, lr=3e-4, **kwargs):
        super(FFNN_clf, self).__init__()
        self.fc_layer_1 = fc_layer_1
        self.fc_layer_2 = fc_layer_2
        self.fc_layer_3 = fc_layer_3
        self.fc_layer_4 = fc_layer_4
        self.lr = lr
        self.criterion = nn.CrossEntropyLoss()

    def construct_network(self, x, y):
        num_inputs = x.shape[1]
        num_outputs = np.unique(y).size
        num_channels = []
        layers = []
        for key in self.__dict__:
            if "layer" in key:
                num_channels += [self.__dict__[key]]
        for i in range(len(num_channels)):
            in_channels = num_inputs if i == 0 else num_channels[i-1]
            out_channels = num_channels[i]
            layers += [nn.Linear(in_channels, out_channels), nn.ReLU()]
        layers += [nn.Linear(num_channels[-1], num_outputs)]
        self.network = nn.Sequential(*layers)

    def fit(self, x, y):
        x = torch.Tensor(x)
        y = torch.Tensor(y).to(torch.int64)
        self.construct_network(x, y)
        self.learn(x, y)

    def predict(self, x):
        x = torch.Tensor(x)
        y_hat = self.forward(x).detach().numpy()
        return y_hat

    def score(self, x, y):
        x = torch.Tensor(x)
        y_hat = self.forward(x).detach().numpy()
        y_hat = np.argmax(y_hat, axis=1)
        score = accuracy_score(y, y_hat)
        return score

class Conv1d_reg(BaseEstimator, Network, nn.Module):

    def __init__(self, conv_layer_1=20, conv_ker_1=30, conv_layer_2=20, conv_ker_2=20, 
                 fc_layer_1=100, fc_layer_2=100, fc_layer_3=50,
                 lr=3e-4, **kwargs):
        super(Conv1d_reg, self).__init__()
        self.conv_layer_1 = conv_layer_1
        self.conv_ker_1 = conv_ker_1
        self.conv_layer_2 = conv_layer_2
        self.conv_ker_2 = conv_ker_2
        self.fc_layer_1 = fc_layer_1
        self.fc_layer_2 = fc_layer_2
        self.fc_layer_3 = fc_layer_3
        self.lr = lr
        self.criterion = nn.MSELoss()

    def construct_network(self, x, y):
        layers = []
        # Structure layer lists
        num_inputs = x.shape[1]
        num_outputs = 1
        conv_channels = []
        for key in self.__dict__:
            if "conv_layer" in key:
                conv_channels += [self.__dict__[key]]
        conv_kers = []
        for key in self.__dict__:
            if "conv_ker" in key:
                conv_kers += [self.__dict__[key]]
        fc_channels = []
        for key in self.__dict__:
            if "fc_layer" in key:
                fc_channels += [self.__dict__[key]]

        # Convolutional layers
        for i in range(len(conv_channels)):
            in_channels = num_inputs if i == 0 else conv_channels[i-1]
            out_channels = conv_channels[i]
            ker_size = conv_kers[i]
            conv = nn.Conv1d(in_channels, out_channels, ker_size)
            layers += [conv, nn.ReLU(), nn.MaxPool1d(2)]
        layers += [nn.Flatten()]
        temp_network = nn.Sequential(*layers)
        x_sample = x[0:1, :]
        flat_channels = temp_network(x_sample).shape[1]
        # Fully connected layers
        for i in range(len(fc_channels)):
            in_channels = flat_channels if i == 0 else fc_channels[i-1]
            out_channels = fc_channels[i]
            layers += [nn.Linear(in_channels, out_channels), nn.ReLU()]
        layers += [nn.Linear(fc_channels[-1], num_outputs)]
        self.network = nn.Sequential(*layers)

    def fit(self, x, y):
        x = torch.Tensor(x)
        y = torch.Tensor(np.expand_dims(y, axis=1))
        self.construct_network(x, y)
        self.learn(x, y)

    def predict(self, x):
        x = torch.Tensor(x)
        y_hat = self.forward(x).detach().numpy()
        y_hat = np.squeeze(y_hat)
        return y_hat

    def score(self, x, y):
        score = r2_score(y, self.predict(x))
        return score


class Conv1d_clf(BaseEstimator, Network, nn.Module):

    def __init__(self, conv_layer_1=10, conv_ker_1=30, conv_layer_2=5, conv_ker_2=20, 
                 fc_layer_1=100, fc_layer_2=100, fc_layer_3=100,
                 lr=3e-4, **kwargs):
        super(Conv1d_clf, self).__init__()
        self.conv_layer_1 = conv_layer_1
        self.conv_ker_1 = conv_ker_1
        self.conv_layer_2 = conv_layer_2
        self.conv_ker_2 = conv_ker_2
        self.fc_layer_1 = fc_layer_1
        self.fc_layer_2 = fc_layer_2
        self.fc_layer_3 = fc_layer_3
        self.lr = lr
        self.criterion = nn.CrossEntropyLoss()

    def construct_network(self, x, y):
        layers = []
        # Structure layer lists
        num_inputs = x.shape[1]
        num_outputs = np.unique(y).size
        conv_channels = []
        for key in self.__dict__:
            if "conv_layer" in key:
                conv_channels += [self.__dict__[key]]
        conv_kers = []
        for key in self.__dict__:
            if "conv_ker" in key:
                conv_kers += [self.__dict__[key]]
        fc_channels = []
        for key in self.__dict__:
            if "fc_layer" in key:
                fc_channels += [self.__dict__[key]]

        # Convolutional layers
        for i in range(len(conv_channels)):
            in_channels = num_inputs if i == 0 else conv_channels[i-1]
            out_channels = conv_channels[i]
            ker_size = conv_kers[i]
            conv = nn.Conv1d(in_channels, out_channels, ker_size)
            layers += [conv, nn.ReLU(), nn.MaxPool1d(2)]
        layers += [nn.Flatten()]
        temp_network = nn.Sequential(*layers)
        x_sample = x[0:1, :]
        flat_channels = temp_network(x_sample).shape[1]
        # Fully connected layers
        for i in range(len(fc_channels)):
            in_channels = flat_channels if i == 0 else fc_channels[i-1]
            out_channels = fc_channels[i]
            layers += [nn.Linear(in_channels, out_channels), nn.ReLU()]
        layers += [nn.Linear(fc_channels[-1], num_outputs)]
        self.network = nn.Sequential(*layers)

    def fit(self, x, y):
        x = torch.Tensor(x)
        y = torch.Tensor(y).to(torch.int64)
        self.construct_network(x, y)
        self.learn(x, y)

    def predict(self, x):
        x = torch.Tensor(x)
        y_hat = self.forward(x).detach().numpy()
        y_hat = np.argmax(y_hat, axis=1)
        return y_hat

    def score(self, x, y):
        self.eval()
        score = accuracy_score(y, self.predict(x))
        return score


class Chomp1d(nn.Module):
    def __init__(self, chomp_size):
        super(Chomp1d, self).__init__()
        self.chomp_size = chomp_size

    def forward(self, x):
        return x[:, :, :-self.chomp_size].contiguous()


class TemporalBlock(nn.Module):
    def __init__(self, n_inputs, n_outputs, kernel_size, stride, dilation, padding, dropout=0.2):
        super(TemporalBlock, self).__init__()
        self.conv1 = weight_norm(nn.Conv1d(n_inputs, n_outputs, kernel_size,
                                           stride=stride, padding=padding, dilation=dilation))
        self.chomp1 = Chomp1d(padding)
        self.relu1 = nn.ReLU()
        self.dropout1 = nn.Dropout(dropout)

        self.conv2 = weight_norm(nn.Conv1d(n_outputs, n_outputs, kernel_size,
                                           stride=stride, padding=padding, dilation=dilation))
        self.chomp2 = Chomp1d(padding)
        self.relu2 = nn.ReLU()
        self.dropout2 = nn.Dropout(dropout)

        self.net = nn.Sequential(self.conv1, self.chomp1, self.relu1, self.dropout1,
                                 self.conv2, self.chomp2, self.relu2, self.dropout2)
        self.downsample = nn.Conv1d(n_inputs, n_outputs, 1) if n_inputs != n_outputs else None
        self.relu = nn.ReLU()
        self.init_weights()

    def init_weights(self):
        self.conv1.weight.data.normal_(0, 0.01)
        self.conv2.weight.data.normal_(0, 0.01)
        if self.downsample is not None:
            self.downsample.weight.data.normal_(0, 0.01)

    def forward(self, x):
        out = self.net(x)
        res = x if self.downsample is None else self.downsample(x)
        return self.relu(out + res)


class TemporalConvNet_reg(BaseEstimator, Network, nn.Module):
    def __init__(self, lr=3e-4, num_tb_channels=[10, 10, 10], num_fc_channels=[200, 200, 100, 100], kernel_size=2, dropout=0.2):
        super(TemporalConvNet_reg, self).__init__()
        self.lr = lr
        self.num_tb_channels = num_tb_channels
        self.num_fc_channels = num_fc_channels
        self.kernel_size = kernel_size
        self.dropout = dropout
        self.criterion = nn.MSELoss()

    def construct_network(self, x, y):
        # Set input and output dimensions (data dependent)
        num_inputs = x.shape[1]
        num_outputs = 1

        layers = []
        # Make temporal blocks
        num_tb_layers = len(self.num_tb_channels)
        for i in range(num_tb_layers):
            dilation_size = 2 ** i
            in_channels = num_inputs if i == 0 else self.num_tb_channels[i-1]
            out_channels = self.num_tb_channels[i]
            layers += [TemporalBlock(in_channels, out_channels, self.kernel_size, stride=1, dilation=dilation_size,
                                     padding=(self.kernel_size-1) * dilation_size, dropout=self.dropout)]
        # Flatten, calculate flattened size, make fully connected blocks
        layers += [nn.Flatten()]
        temp_network = nn.Sequential(*layers)
        x_sample = x[0:1, :]
        flat_channels = temp_network(x_sample).shape[1]
        num_fc_layers = len(self.num_fc_channels)
        for i in range(num_fc_layers):
            in_channels = flat_channels if i == 0 else self.num_fc_channels[i-1]
            out_channels = self.num_fc_channels[i]
            layers += [nn.Linear(in_channels, out_channels), nn.ReLU()]
        layers += [nn.Linear(self.num_fc_channels[-1], num_outputs)]
        self.network = nn.Sequential(*layers)


    def fit(self, x, y):
        x = torch.Tensor(x)
        y = torch.Tensor(np.expand_dims(y, axis=1))
        self.construct_network(x, y)
        self.learn(x, y)

    def predict(self, x):
        x = torch.Tensor(x)
        y_hat = self.forward(x).detach().numpy()
        y_hat = np.squeeze(y_hat)
        return y_hat

    def score(self, x, y):
        score = r2_score(y, self.predict(x))
        return score


class TemporalConvNet_clf(BaseEstimator, Network, nn.Module):
    def __init__(self, lr=3e-3, num_tb_channels=[20, 20], num_fc_channels=[500, 200, 100, 50], kernel_size=2, dropout=0.2):
        super(TemporalConvNet_clf, self).__init__()
        self.lr = lr
        self.num_tb_channels = num_tb_channels
        self.num_fc_channels = num_fc_channels
        self.kernel_size = kernel_size
        self.dropout = dropout
        self.criterion = nn.CrossEntropyLoss()

    def construct_network(self, x, y):
        # Set input and output dimensions (data dependent)
        num_inputs = x.shape[1]
        num_outputs = np.unique(y).size

        layers = []
        # Make temporal blocks
        num_tb_layers = len(self.num_tb_channels)
        for i in range(num_tb_layers):
            dilation_size = 2 ** i
            in_channels = num_inputs if i == 0 else self.num_tb_channels[i-1]
            out_channels = self.num_tb_channels[i]
            layers += [TemporalBlock(in_channels, out_channels, self.kernel_size, stride=1, dilation=dilation_size,
                                     padding=(self.kernel_size-1) * dilation_size, dropout=self.dropout)]
        # Flatten, calculate flattened size, make fully connected blocks
        layers += [nn.Flatten()]
        temp_network = nn.Sequential(*layers)
        x_sample = x[0:1, :]
        flat_channels = temp_network(x_sample).shape[1]
        num_fc_layers = len(self.num_fc_channels)
        for i in range(num_fc_layers):
            in_channels = flat_channels if i == 0 else self.num_fc_channels[i-1]
            out_channels = self.num_fc_channels[i]
            layers += [nn.Linear(in_channels, out_channels), nn.ReLU()]
        layers += [nn.Linear(self.num_fc_channels[-1], num_outputs)]
        self.network = nn.Sequential(*layers)


    def fit(self, x, y):
        x = torch.Tensor(x)
        y = torch.Tensor(y).to(torch.int64)
        self.construct_network(x, y)
        self.learn(x, y)

    def predict(self, x):
        x = torch.Tensor(x)
        y_hat = self.forward(x).detach().numpy()
        y_hat = np.argmax(y_hat, axis=1)
        return y_hat

    def score(self, x, y):
        self.eval()
        score = accuracy_score(y, self.predict(x))
        return score