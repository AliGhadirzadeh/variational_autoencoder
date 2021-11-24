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


class PredictorNetwork(nn.Module):

    def __init__(self, **kwargs):
        super(PredictorNetwork, self).__init__()
        self.network = None
        self.type = None

    def construct_network(self, x, y):
        self.set_type(y)
        n_inputs = x.shape[1]
        if self.type == "categorical":
            self.network = nn.Sequential(nn.Linear(n_inputs, 100), nn.BatchNorm1d(100), nn.ReLU(),
                                         nn.Linear(100, 100), nn.BatchNorm1d(100), nn.ReLU(),
                                         nn.Linear(100, 100), nn.BatchNorm1d(100), nn.ReLU(),
                                         nn.Linear(100, self.n_outputs), nn.Softmax(dim=1))
        elif self.type == "continuous":
            self.network = nn.Sequential(nn.Linear(n_inputs, 100), nn.BatchNorm1d(100), nn.ReLU(),
                                         nn.Linear(100, 100), nn.BatchNorm1d(100), nn.ReLU(),
                                         nn.Linear(100, 100), nn.BatchNorm1d(100), nn.ReLU(),
                                         nn.Linear(100, self.n_outputs))

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


class MarginalNetwork(nn.Module):

    def __init__(self, **kwargs):
        super(MarginalNetwork, self).__init__()
        self.network = None

    def construct_network(self, x):
        n_inputs = x.shape[1]
        dropout = 0.2
        self.network = nn.Sequential(nn.Linear(n_inputs, 300), nn.BatchNorm1d(300), nn.ReLU(),
                                     nn.Linear(300, 300), nn.BatchNorm1d(300), nn.ReLU(), 
                                     nn.Linear(300, 300), nn.BatchNorm1d(300), nn.ReLU(), 
                                     nn.Linear(300, 300), nn.BatchNorm1d(300), nn.ReLU(), 
                                     nn.Linear(300, 300), nn.BatchNorm1d(300), nn.ReLU(),
                                     nn.Linear(300, 300), nn.BatchNorm1d(300), nn.ReLU(), 
                                     nn.Linear(300, n_inputs))

    def forward(self, x):
        return self.network(x)


class ExtractorNetwork(nn.Module):

    def __init__(self):
        super(ExtractorNetwork, self).__init__()
        self.network = None

    def construct_network(self, x):
        f_1 = 8
        f_2 = 16
        f_t = [12]
        k_e = 32
        k_t = 4
        p_e = 0.2
        p_t = 0.2

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
                               TemporalConvNet(f_2, f_t, kernel_size=k_t, dropout=p_t))

    def forward(self, x):
        x = torch.unsqueeze(x, dim=1)
        return self.network(x)


class AdvLatentNet(nn.Module, BaseEstimator):

    def __init__(self, alpha=0.9):
        super(AdvLatentNet, self).__init__()
        self.alpha = alpha
        self.extr_lr = 3e-4
        self.marg_lr = 8e-5

        self.extr_net = ExtractorNetwork()

        self.aux_net_y = PredictorNetwork()
        self.aux_net_c = PredictorNetwork()

        self.marg_net = MarginalNetwork()

        self.pred_net_y = PredictorNetwork()
        self.pred_net_c = PredictorNetwork()

        self.probe_net = PredictorNetwork()

        self.network_list = [self.extr_net, self.aux_net_y, self.aux_net_c,
                             self.marg_net, self.pred_net_y, self.pred_net_c,
                             self.probe_net]

        self.extr_epoch = 400
        self.marg_epoch = 200
        self.probe_epoch = 200
        
        self.tqdm_disable = False
        self.writer = None

    def construct_network(self, data):
        with torch.no_grad():
            x, y, c = data[:]
            self.extr_net.construct_network(x)
            z = self.extr_net(x)
            self.aux_net_y.construct_network(z, y)
            self.aux_net_c.construct_network(z, c)
            self.marg_net.construct_network(z)
            z_marg = self.marg_net(z)
            self.pred_net_y.construct_network(z_marg, y)
            self.pred_net_c.construct_network(z_marg, c)
            self.probe_net.construct_network(z_marg, c)

    def forward(self, x):
        z = self.extr_net(x)
        z_marg = self.marg_net(z)
        y = self.pred_net_y(z_marg)
        c = self.pred_net_c(z_marg)
        return y, c

    def forward_aux(self, x):
        z_marg = self.forward_extr(x)
        y = self.aux_net_y(z_marg)
        c = self.aux_net_c(z_marg)
        return y, c

    def forward_extr(self, x):
        z = self.extr_net(x)
        return z

    def forward_marg(self, z):
        z_marg = self.marg_net(z)
        y = self.pred_net_y(z_marg)
        c = self.pred_net_c(z_marg)
        return y, c

    def fit(self, data, save_path=None, load_path=None):
        train_data, val_data = data.train_test_split()
        train_loader = DataLoader(train_data, batch_size=32, shuffle=True)

        if load_path:
            print("Loading pre-trained feature extractor")
            self.extr_net.load_state_dict(torch.load(load_path))
        else:
            # Train feature extraction
            extr_opt = optim.Adam(self.parameters(), lr=self.extr_lr)
            self.activate([self.extr_net, self.aux_net_y, self.aux_net_c])

            for epoch in tqdm(range(self.extr_epoch), disable=self.tqdm_disable):
                for b_x, b_y, b_c in train_loader:
                    self.train()
                    extr_opt.zero_grad()
                    b_y_hat, b_c_hat = self.forward_aux(b_x)
                    y_loss = self.aux_net_y.loss(b_y_hat, b_y)
                    c_loss = self.aux_net_c.loss(b_c_hat, b_c)
                    extr_loss = y_loss + c_loss
                    extr_loss.backward()
                    extr_opt.step()

                with torch.no_grad():
                    self.eval()
                    self.evaluate_extr(train_data, val_data, epoch)
            if save_path:
                torch.save(self.extr_net.state_dict(), save_path)

        # Train latent manipulations
        marg_opt = optim.Adam(self.marg_net.parameters(), lr=self.marg_lr)
        y_opt = optim.Adam(self.pred_net_y.parameters(), lr=self.marg_lr)
        c_opt = optim.Adam(self.pred_net_c.parameters(), lr=self.marg_lr)

        self.activate([self.marg_net, self.pred_net_y, self.pred_net_c])

        for epoch in tqdm(range(self.marg_epoch), disable=self.tqdm_disable):

            for b_x, b_y, b_c in train_loader:
                self.train()
                  
                y_opt.zero_grad()
                c_opt.zero_grad()
                b_y_hat, b_c_hat = self.forward(b_x)
                y_loss = self.pred_net_y.loss(b_y_hat, b_y)
                c_loss = self.pred_net_c.loss(b_c_hat, b_c)
                y_loss.backward(retain_graph=True)
                c_loss.backward()
                c_opt.step()
                y_opt.step()

                marg_opt.zero_grad()
                b_y_hat, b_c_hat = self.forward(b_x)
                y_loss = self.pred_net_y.loss(b_y_hat, b_y)
                c_loss = self.pred_net_c.loss(b_c_hat, b_c)
                marg_loss = y_loss - self.alpha*c_loss

                marg_loss.backward()
                marg_opt.step()
                
            with torch.no_grad():
                self.eval()
                self.evaluate_marg(train_data, val_data, epoch)

    def activate(self, networks):
        for net in self.network_list:
            if net in networks:
                for param in net.parameters():
                    param.requires_grad = True
            else:
                for param in net.parameters():
                    param.requires_grad = False
                
    def marg_loss(self, y_hat, y, c_hat, c):
        y_loss = self.pred_net_y.loss(y_hat, y)
        c_loss = self.pred_net_c.loss(c_hat, c)
        return y_loss - self.alpha*c_loss

    def evaluate_extr(self, train_data, test_data, epoch):
        train_x, train_y, train_c = train_data[:]
        test_x, test_y, test_c = test_data[:]


        train_y_hat, train_c_hat = self.forward_aux(train_x)
        test_y_hat, test_c_hat = self.forward_aux(test_x)

        y_loss = self.aux_net_y.loss(test_y_hat, test_y)
        c_loss = self.aux_net_c.loss(test_c_hat, test_c)

        train_y_score = self.aux_net_y.score(train_y_hat, train_y)
        train_c_score = self.aux_net_c.score(train_c_hat, train_c)
        test_y_score = self.aux_net_y.score(test_y_hat, test_y)
        test_c_score = self.aux_net_c.score(test_c_hat, test_c)

        print("Train score (y): " + "{:+.3f}".format(train_y_score))
        print("Test score (y):  " + "{:+.3f}".format(test_y_score))
        print()
        print("Train score (c): " + "{:+.3f}".format(train_c_score))
        print("Test score (c):  " + "{:+.3f}".format(test_c_score))

        if self.writer:
            self.writer.add_scalars("Loss", {"y_aux" : y_loss,
                                             "c_aux" : c_loss}, 
                                             epoch)
            self.writer.add_scalars("Score/y_aux", {"train" : train_y_score, 
                                                    "test" : test_y_score},
                                                    epoch)
            self.writer.add_scalars("Score/c_aux", {"train" : train_c_score,
                                                    "test" : test_c_score},
                                                    epoch)
    
    def evaluate_marg(self, train_data, test_data, epoch):
        train_x, train_y, train_c = train_data[:]
        test_x, test_y, test_c = test_data[:]


        train_y_hat, train_c_hat = self.forward(train_x)
        test_y_hat, test_c_hat = self.forward(test_x)

        marg_loss = self.marg_loss(test_y_hat, test_y, test_c_hat, test_c)
        y_loss = self.pred_net_y.loss(test_y_hat, test_y)
        c_loss = self.pred_net_c.loss(test_c_hat, test_c)

        train_y_score = self.pred_net_y.score(train_y_hat, train_y)
        train_c_score = self.pred_net_c.score(train_c_hat, train_c)
        test_y_score = self.pred_net_y.score(test_y_hat, test_y)
        test_c_score = self.pred_net_c.score(test_c_hat, test_c)

        print("Train score (y): " + "{:+.3f}".format(train_y_score))
        print("Test score (y):  " + "{:+.3f}".format(test_y_score))
        print()
        print("Train score (c): " + "{:+.3f}".format(train_c_score))
        print("Test score (c):  " + "{:+.3f}".format(test_c_score))

        if self.writer:
            self.writer.add_scalars("Loss", {"marg" : marg_loss,
                                             "y_pred" : y_loss,
                                             "c_pred" : c_loss}, 
                                             epoch)
            self.writer.add_scalars("Score/y_pred", {"train" : train_y_score, 
                                                            "test" : test_y_score},
                                                            epoch)
            self.writer.add_scalars("Score/c_pred", {"train" : train_c_score,
                                                             "test" : test_c_score},
                                                              epoch)
            if epoch == self.marg_epoch-1:
                z = self.forward_extr(test_x)
                self.writer.add_embedding(z)
                self.writer.add_graph(self, input_to_model=test_x)
                self.writer.close()

    def evaluate_probe(self, train_data, test_data, epoch):
        train_x, train_y, train_c = train_data[:]
        test_x, test_y, test_c = test_data[:]

        train_z = self.forward_extr(train_x)
        train_z_marg = self.marg_net(train_z)
        train_c_hat = self.probe_net(train_z_marg)

        test_z = self.forward_extr(test_x)
        test_z_marg = self.marg_net(test_z)
        test_c_hat = self.probe_net(test_z_marg)

        probe_loss = self.probe_net.loss(test_c_hat, test_c)

        train_c_score = self.probe_net.score(train_c_hat, train_c)
        test_c_score = self.probe_net.score(test_c_hat, test_c)

        print("Train score (c): " + "{:+.3f}".format(train_c_score))
        print("Test score (c):  " + "{:+.3f}".format(test_c_score))

        if self.writer:
            self.writer.add_scalars("Loss", {"probe" : probe_loss}, 
                                             epoch)
            self.writer.add_scalars("Score/probe", {"train" : train_c_score,
                                                    "test" : test_c_score},
                                                    epoch)

    def probe(self, data):
        train_data, val_data = data.train_test_split()
        train_loader = DataLoader(train_data, batch_size=32, shuffle=True)

        # TRAIN FEATURE EXTRACTION
        probe_opt = optim.Adam(self.probe_net.parameters(), lr=3e-4)
        self.activate([self.probe_net])
        for epoch in tqdm(range(self.probe_epoch), disable=self.tqdm_disable):
            for b_x, b_y, b_c in train_loader:
                self.train()
                probe_opt.zero_grad()
                b_z = self.forward_extr(b_x)
                b_z_marg = self.marg_net(b_z)
                b_c_hat = self.probe_net(b_z_marg)
                probe_loss = self.probe_net.loss(b_c_hat, b_c)
                probe_loss.backward()
                probe_opt.step()

            with torch.no_grad():
                self.eval()
                self.evaluate_probe(train_data, val_data, epoch)


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