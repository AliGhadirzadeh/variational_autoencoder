import numpy as np
import pandas as pd
from tqdm import tqdm 

import utils
import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader
from torch.utils.tensorboard import SummaryWriter
import torch.optim as optim

from sklearn.metrics import r2_score, accuracy_score



class Dataset(Dataset):

    def __init__(self, x, df):
        self.x = x
        self.df = df
        self.dep_variable = None
        self.indep_variable = None
        self.ctrl_variable = None

    def __len__(self):
        return self.df.shape[0]

    def __getitem__(self, index):
        indep_sample = self.x[index]
        dep_sample = torch.as_tensor(self.df[self.dep_variable][index])
        ctrl_sample = torch.as_tensor(self.df[self.ctrl_variable][index])
        return indep_sample, dep_sample, ctrl_sample

    def train_test_split(self, test_size=0.2):
        train_size = int((1-test_size) * len(self))
        test_size = len(self) - train_size
        return torch.utils.data.random_split(self, [train_size, test_size])

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
        self.conv = False

    def construct_network(self, x):
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


class ControlledNetwork(nn.Module):
    """docstring for CompositeNetwork"""
    def __init__(self, alpha=1, beta=1):
        super(ControlledNetwork, self).__init__()
        self.extractor = ExtractorNetwork()
        self.dep_predictor = PredictorNetwork()
        self.ctrl_predictor = PredictorNetwork()

        self.n_epoch = 100
        self.lr = 3e-4

        self.writer = SummaryWriter()


    def construct_network(self, data):
        with torch.no_grad():
            x, y, c = data[:]
            self.extractor.construct_network(x)
            x_latent = self.extractor(x)
            self.dep_predictor.construct_network(x_latent, y)
            self.ctrl_predictor.construct_network(x_latent, c)

    def forward(self, x):
        latent_x = self.extractor(x)
        y = self.dep_predictor(latent_x)
        c = self.ctrl_predictor(latent_x)
        return y, c

    def learn(self, data):
        train_data, val_data = data.train_test_split()
        train_loader = DataLoader(train_data, batch_size=32, shuffle=True)
        val_loader = DataLoader(val_data, batch_size=32, shuffle=True)

        # One optimizer per network
        extractor_opt = optim.Adam(self.parameters(), lr=self.lr)
        dep_predictor_opt = optim.Adam(self.parameters(), lr=self.lr)
        ctrl_predictor_opt = optim.Adam(self.parameters(), lr=self.lr)

        for epoch in tqdm(range(self.n_epoch)):

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
                self.evaluate(val_loader, epoch)

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
        extractor_loss = dep_loss - 0.5 * ctrl_loss
        return extractor_loss
    
    def evaluate(self, test_loader, epoch):
        n_batches = len(test_loader)
        i = 0
        extractor_loss = np.zeros(n_batches)
        dep_loss = np.zeros(n_batches)
        ctrl_loss = np.zeros(n_batches)

        dep_score = np.zeros(n_batches)
        ctrl_score = np.zeros(n_batches)

        for b_x, b_y, b_c in test_loader:
            b_y_hat, b_c_hat = self.forward(b_x)
            extractor_loss[i] = self.extractor_loss(b_y_hat, b_y, b_c_hat, b_c)
            dep_loss[i] = self.dep_predictor.loss(b_y_hat, b_y)
            ctrl_loss[i] += self.ctrl_predictor.loss(b_c_hat, b_c)
            dep_score[i] = self.dep_predictor.score(b_y_hat, b_y)
            ctrl_score[i] = self.ctrl_predictor.score(b_c_hat, b_c)
            i += 1

        self.writer.add_scalars("Loss", {"extractor" : extractor_loss.mean(),
                                         "dep_predictor" : dep_loss.mean(),
                                         "ctrl_predictor" : ctrl_loss.mean()}, 
                                         epoch)
        self.writer.add_scalar('Accuracy/dep_predictor', dep_score.mean(), epoch)
        self.writer.add_scalar('Accuracy/ctrl_predictor', ctrl_score.mean(), epoch)

        """
        print("Extractor loss (validation): %.4f" % extractor_loss.mean())
        print("Dependent variable loss (validation): %.4f" % dep_loss.mean())
        print("Control variable loss (validation): %.4f" % ctrl_loss.mean())

        print("Dependent variable score (validation): %.4f" % dep_score.mean())
        print("Control variable score (validation): %.4f" % ctrl_score.mean())
        """




# EEG
"""
x, y = utils.get_data("EEG", "subject_id")
x = torch.Tensor(x)
y = torch.Tensor(y)
df = pd.read_pickle("../../data/data/numpy_files/df.pkl")

data = Dataset(x, df)
data.indep_variable = "EEG"
data.dep_variable = "math_t1"
data.ctrl_variable = "subject_id"
"""

# Digits

x, y = utils.get_data("digits", "subject_id")
x = torch.Tensor(x)
df = pd.DataFrame(y, columns=["target"])

data = Dataset(x, df)
data.dep_variable = "target"
data.indep_variable = "digits"
data.ctrl_variable = "target"


cn = ControlledNetwork()
cn.construct_network(data)
cn.learn(data)
