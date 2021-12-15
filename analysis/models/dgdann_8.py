import numpy as np
from tqdm import tqdm 

import torch
import torch.nn as nn
from torch.nn.utils import weight_norm
from torch.utils.data import DataLoader
from torch.utils.tensorboard import SummaryWriter
import torch.optim as optim
from torch.optim.lr_scheduler import MultiStepLR

from sklearn.base import BaseEstimator
from sklearn.metrics import r2_score, accuracy_score

# https://discuss.pytorch.org/t/solved-reverse-gradients-in-backward-pass/3589/4
# Grad rev layer


class Gf_net(nn.Module):

    def __init__(self):
        super(Gf_net, self).__init__()
        f_1 = 8
        f_2 = 16
        f_t = [12]
        k_e = 32
        k_t = 4
        p_e = 0.2
        p_t = 0.2
        n_channels = 8

        self.network = nn.Sequential(
                               # Temporal convolution
                               nn.Conv2d(1, f_1, (1, k_e), padding="same"),
                               nn.BatchNorm2d(f_1),

                               # Depthwise convolution
                               DepthwiseConv2d(f_1, 2*f_1, kernel_size=(n_channels, 1)),
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
                               TemporalConvNet(f_2, f_t, kernel_size=k_t, dropout=p_t))


    def forward(self, x):
        x = torch.unsqueeze(x, dim=1)
        return self.network(x)


class Gy_net(nn.Module):

    def __init__(self):
        super(Gy_net, self).__init__()
        n_input = 12
        n_output = 1
        self.network = nn.Sequential(nn.Linear(n_input, n_output))
        self.criterion = nn.MSELoss()

    def forward(self, x):
        return self.network(x)

    def loss(self, y_hat, y):
        y = y.view((y.shape[0], 1))
        return self.criterion(y_hat, y)

    def score(self, y_hat, y):
        return r2_score(y, y_hat)


class Gd_net(nn.Module):

    def __init__(self):
        super(Gd_net, self).__init__()
        n_input = 12
        n_output = 42
        self.network = nn.Sequential(nn.Linear(n_input, n_output))
        self.criterion = nn.CrossEntropyLoss()

    def forward(self, x):
        return self.network(x)

    def loss(self, d_hat, d):
        return self.criterion(d_hat, d)

    def score(self, d_hat, d):
        d_hat = np.argmax(d_hat, axis=1)
        return accuracy_score(d, d_hat)


class Probe_net(nn.Module):

    def __init__(self):
        super(Probe_net, self).__init__()
        n_input = 12
        n_output = 42
        self.network = nn.Sequential(nn.Linear(n_input, 100), nn.BatchNorm1d(100), nn.ReLU(),
                                     nn.Linear(100, 100), nn.BatchNorm1d(100), nn.ReLU(),
                                     nn.Linear(100, 100), nn.BatchNorm1d(100), nn.ReLU(),
                                     nn.Linear(100, n_output))
        self.criterion = nn.CrossEntropyLoss()

    def forward(self, x):
        return self.network(x)

    def loss(self, d_hat, d):
        return self.criterion(d_hat, d)

    def score(self, d_hat, d):
        d_hat = np.argmax(d_hat, axis=1)
        return accuracy_score(d, d_hat)



class DG_DANN_8(nn.Module, BaseEstimator):

    def __init__(self, alpha=0.0):
        super(DG_DANN_8, self).__init__()
        self.alpha = alpha
        self.lr = 5e-4

        self.Gf = Gf_net()
        self.Gy = Gy_net()
        self.Gd = Gd_net()

        self.n_epoch = 50
        
        self.tqdm_disable = False
        self.writer = None
        self.print = False

        self.best_score = 0

    def forward(self, x):
        z = self.Gf(x)
        y = self.Gy(z)
        reverse_z = grad_reverse(z)
        d = self.Gd(reverse_z)
        return y, d

    def fit(self, data):
        train_data, val_data = data.train_test_split()
        train_loader = DataLoader(train_data, batch_size=8, shuffle=True)
        params = list(self.Gf.parameters()) + list(self.Gy.parameters()) + list(self.Gd.parameters())

        opt = optim.Adam(params, lr=self.lr)

        for epoch in tqdm(range(self.n_epoch), disable=self.tqdm_disable):

            for b_x, b_y, b_d in train_loader:
                self.train()
                opt.zero_grad()
                b_y_hat, b_d_hat = self.forward(b_x)
                Gy_loss = self.Gy.loss(b_y_hat, b_y)
                Gd_loss = self.Gd.loss(b_d_hat, b_d)
                loss = Gy_loss - self.alpha*Gd_loss
                loss.backward()
                opt.step()
                
            with torch.no_grad():
                self.eval()
                self.evaluate(train_data, val_data, epoch)

                
    def evaluate(self, train_data, test_data, epoch):
        train_x, train_y, train_d = train_data[:]
        test_x, test_y, test_d = test_data[:]


        train_y_hat, train_d_hat = self.forward(train_x)
        test_y_hat, test_d_hat = self.forward(test_x)

        train_Gy_loss = self.Gy.loss(train_y_hat, train_y)
        train_Gd_loss = self.Gd.loss(train_d_hat, train_d)
        train_loss = train_Gy_loss - self.alpha*train_Gd_loss

        test_Gy_loss = self.Gy.loss(test_y_hat, test_y)
        test_Gd_loss = self.Gd.loss(test_d_hat, test_d)
        test_loss = test_Gy_loss - self.alpha*test_Gd_loss

        train_Gy_score = self.Gy.score(train_y_hat, train_y)
        train_Gd_score = self.Gd.score(train_d_hat, train_d)

        test_Gy_score = self.Gy.score(test_y_hat, test_y)
        test_Gd_score = self.Gd.score(test_d_hat, test_d)

        if test_Gy_score > self.best_score:
            self.best_score = test_Gy_score

        if self.print:
            print("Train score (Gy): " + "{:+.3f}".format(train_Gy_score))
            print("Test score (Gy):  " + "{:+.3f}".format(test_Gy_score))
            print()
            print("Train score (Gd): " + "{:+.3f}".format(train_Gd_score))
            print("Test score (Gd):  " + "{:+.3f}".format(test_Gd_score))

        if self.writer:
            self.writer.add_scalars("Loss", {"train_Gy" : train_Gy_loss,
                                             "test_Gy" : test_Gy_loss,
                                             "train_Gd" : train_Gd_loss,
                                             "test_Gd" : test_Gd_loss,
                                             "train_loss" : train_loss,
                                             "test_loss" : test_loss}, 
                                             epoch)
            self.writer.add_scalars("Score", {"train_Gy" : train_Gy_score, 
                                              "test_Gy" : test_Gy_score,
                                              "train_Gd" : train_Gd_score,
                                              "test_Gd" : test_Gd_score},
                                              epoch)
            if epoch == self.n_epoch-1:
                z = self.Gf(test_x)
                self.writer.add_embedding(z)
                self.writer.add_graph(self, input_to_model=test_x)
                self.writer.close()

    def probe(self, data):
        train_data, val_data = data.train_test_split()
        train_loader = DataLoader(train_data, batch_size=32, shuffle=True)

        probe = Probe_net()
        probe.lr = 3e-4
        probe.epoch = 50
        opt = optim.Adam(probe.parameters(), lr=probe.lr)

        for epoch in tqdm(range(probe.epoch), disable=self.tqdm_disable):
            for b_x, b_y, b_d in train_loader:
                self.train()
                opt.zero_grad()
                b_z = self.Gf(b_x)
                b_d_hat = probe(b_z)
                loss = probe.loss(b_d_hat, b_d)
                loss.backward()
                opt.step()

            with torch.no_grad():
                self.eval()
                self.evaluate_probe(probe, train_data, val_data, epoch)

    def evaluate_probe(self, probe, train_data, test_data, epoch):
        train_x, train_y, train_d = train_data[:]
        test_x, test_y, test_d = test_data[:]

        train_z = self.Gf(train_x)
        train_d_hat = probe(train_z)

        test_z = self.Gf(test_x)
        test_d_hat = probe(test_z)

        train_loss = probe.loss(train_d_hat, train_d)
        test_loss = probe.loss(test_d_hat, test_d)

        train_score = probe.score(train_d_hat, train_d)
        test_score = probe.score(test_d_hat, test_d)

        if self.print:
            print("Train score (probe): " + "{:+.3f}".format(train_score))
            print("Test score (probe):  " + "{:+.3f}".format(test_score))

        if self.writer:
            self.writer.add_scalars("Loss", {"train_probe" : train_loss,
                                             "test_probe" : test_loss}, 
                                             epoch)
            self.writer.add_scalars("Score", {"train_probe" : train_score,
                                              "test_probe" : test_score},
                                              epoch)


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

from torch.autograd import Function
class GradReverse(Function):
    @staticmethod
    def forward(ctx, x):
        return x.view_as(x)

    @staticmethod
    def backward(ctx, grad_output):
        return grad_output.neg()

def grad_reverse(x):
    return GradReverse.apply(x)