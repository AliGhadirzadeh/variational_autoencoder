import numpy as np
from tqdm import tqdm 

import torch
import torch.nn as nn
from torch.nn.utils import weight_norm
from torch.utils.data import DataLoader
from torch.utils.tensorboard import SummaryWriter
import torch.optim as optim

from sklearn.base import BaseEstimator
from sklearn.metrics import r2_score, accuracy_score


class EEGTCN(nn.Module, BaseEstimator):

    def __init__(self, lr=3e-4):
        super(EEGTCN, self).__init__()
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
            layers = []

            f_1 = 8
            f_2 = 16
            f_t = [12]
            k_e = 32
            k_t = 4
            p_e = 0.2
            p_t = 0.3

            self.network = nn.Sequential(
                               # Temporal convolution
                               nn.Conv2d(1, f_1, (1, k_e), padding="same"),
                               nn.BatchNorm2d(f_1),

                               # Depthwise convolution
                               DepthwiseConv2d(f_1, 2*f_1, kernel_size=(20, 1)),
                               nn.BatchNorm2d(2*f_1),
                               nn.ELU(),
                               nn.AvgPool2d((1, 8)),
                               nn.Dropout(p_e),

                               # Separable convolution
                               SeparableConv2d(2*f_1, f_2, (1, 16)),
                               nn.BatchNorm2d(f_2),
                               nn.ELU(),
                               nn.AvgPool2d((1, 8)),
                               nn.Dropout(p_e),

                               # TCN
                               TemporalConvNet(f_2, f_t, k_t),

                               # FC
                               nn.Linear(f_t[0], self.n_outputs))


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
        x = torch.unsqueeze(x, dim=1)
        for layer in self.network:
            x = layer(x)
        return x

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

        print(train_score)
        print(test_score)

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
                extractor = self.network[:-1]
                latent_x = extractor(torch.unsqueeze(test_x, dim=1))
                self.writer.add_embedding(latent_x)
                self.writer.add_graph(self, input_to_model=test_x)
                self.writer.close()


class SeparableConv2d(nn.Module):

    def __init__(self, in_channels, out_channels, kernel_size, bias=False):
        super(SeparableConv2d, self).__init__()
        self.depthwise = nn.Conv2d(in_channels, in_channels, kernel_size=kernel_size, 
                                   groups=in_channels, bias=bias, padding="same")
        self.pointwise = nn.Conv2d(in_channels, out_channels, 
                                   kernel_size=1, bias=bias, padding="same")

    def forward(self, x):
        x = self.depthwise(x)
        x = self.pointwise(x)
        return x


class DepthwiseConv2d(nn.Module):

    def __init__(self, in_channels, out_channels, kernel_size, bias=False):
        super(DepthwiseConv2d, self).__init__()
        self.depthwise = nn.Conv2d(in_channels, out_channels, kernel_size=kernel_size, 
                                   groups=in_channels, bias=bias, padding=0)

    def forward(self, x):
        x = self.depthwise(x)
        return x


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


class TemporalConvNet(nn.Module):
    def __init__(self, num_inputs, num_channels, kernel_size=2, dropout=0.2):
        super(TemporalConvNet, self).__init__()
        layers = []
        num_levels = len(num_channels)
        for i in range(num_levels):
            dilation_size = 2 ** i
            in_channels = num_inputs if i == 0 else num_channels[i-1]
            out_channels = num_channels[i]
            layers += [TemporalBlock(in_channels, out_channels, kernel_size, stride=1, dilation=dilation_size,
                                     padding=(kernel_size-1) * dilation_size, dropout=dropout)]

        self.network = nn.Sequential(*layers)

    def forward(self, x):
        x = torch.squeeze(x)
        x = self.network(x)
        return x[:, :, -1]