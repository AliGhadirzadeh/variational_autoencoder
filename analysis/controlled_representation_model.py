import numpy as np
import pandas as pd
from tqdm import tqdm 

import utils
import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader
import torch.optim as optim

from sklearn.metrics import r2_score, accuracy_score



class Dataset(Dataset):

    def __init__(self, x, df, target_string=None):
        self.x = x
        self.df = df
        self.dep_variable = None
        self.indep_variable = None
        self.ctrl_variables = []

    def __len__(self):
        return self.df.shape[0]

    def __getitem__(self, index):
        dep_sample = self.x[index]
        indep_sample = torch.as_tensor(self.df[self.indep_variable][index])
        ctrl_sample = {}
        for var in self.ctrl_variables:
            ctrl_sample[var] = torch.as_tensor(self.df[var][index])
        return dep_sample, indep_sample, ctrl_sample

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
        num_channels = 50
        num_outputs = None

        if self.type == "categorical":
            self.criterion = nn.CrossEntropyLoss()
            num_outputs = torch.unique(y).shape[0]
        elif self.type == "continuous":
            self.criterion = nn.MSELoss()
            num_outputs = 1
        else:
            raise

        layers = [nn.Linear(num_inputs, num_channels), nn.ReLU()]
        layers += [nn.Linear(num_channels, num_outputs)]
        
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

    def construct_network(self, x):
        num_inputs = x.shape[1]
        num_channels = 50
        num_outputs = 50
        layers = [nn.Linear(num_inputs, num_channels), nn.ReLU()]
        layers += [nn.Linear(num_channels, num_outputs), nn.ReLU()]
        self.network = nn.Sequential(*layers)

    def forward(self, x):
        return self.network(x)

    def loss(self, x_hat, x):
        return self.criterion(x_hat, x)


class ControlledNetwork(nn.Module):
    """docstring for CompositeNetwork"""
    def __init__(self, alpha=1, beta=1):
        super(ControlledNetwork, self).__init__()
        self.n_epoch = 100

    def construct_network(self, data):
        with torch.no_grad():
            x, y, c = data[:]

            self.extractor = ExtractorNetwork()
            self.extractor.construct_network(x)
            x_latent = self.extractor(x)

            self.dep_predictor = PredictorNetwork()
            self.dep_predictor.construct_network(x_latent, y)


            self.ctrl_predictors = {}
            for var in c.keys():
                self.ctrl_predictors[var] = PredictorNetwork()
                self.ctrl_predictors[var].construct_network(x_latent, c[var])

    def forward(self, x):
        latent_x = self.extractor(x)
        y = self.dep_predictor(latent_x)
        c = {}
        for var in self.ctrl_predictors.keys():
            c[var] = self.ctrl_predictors[var](latent_x)
        return y, c

    def learn(self, data):
        train_data, val_data = data.train_test_split()
        train_loader = DataLoader(train_data, batch_size=32, shuffle=True)
        val_loader = DataLoader(val_data, batch_size=32, shuffle=True)
        opt = optim.Adam(self.parameters(), lr=3e-4)
        for epoch in tqdm(range(self.n_epoch)):
            for b_x, b_y, b_c in train_loader:
                self.train()
                opt.zero_grad()
                b_y_hat, b_c_hat = self.forward(b_x)
                loss_tuple = self.loss(b_y_hat, b_y, b_c_hat, b_c)
                self.update(loss_tuple)
                opt.step()
            with torch.no_grad():
                iterations = 0
                extractor_loss = 0
                dep_loss = 0
                ctrl_loss = 0

                dep_score = 0
                ctrl_score = 0
                for b_x, b_y, b_c in val_loader:
                    b_y_hat, b_c_hat = self.forward(b_x)
                    loss_tuple = self.loss(b_y_hat, b_y, b_c_hat, b_c)
                    extractor_loss += loss_tuple[0]
                    dep_loss += loss_tuple[1]
                    ctrl_loss += loss_tuple[2]["target"]

                    score_tuple = self.score(b_y_hat, b_y, b_c_hat, b_c)
                    dep_score += score_tuple[0]
                    ctrl_score += score_tuple[1]["target"]

                    iterations += 1

                extractor_loss /= iterations
                dep_loss /= iterations
                ctrl_loss /= iterations

                dep_score /= iterations
                ctrl_score /= iterations

                print("Extractor loss (validation): %.4f" % extractor_loss)
                print("Dependent variable loss (validation): %.4f" % dep_loss)
                print("Control variable loss (validation): %.4f" % ctrl_loss)
                print("Dependent variable score (validation): %.4f" % dep_score)
                print("Control variable score (validation): %.4f" % ctrl_score)

    def update(self, loss_tuple):
        extractor_loss, dep_loss, ctrl_loss = loss_tuple
        extractor_loss.backward(retain_graph=True)
        dep_loss.backward(retain_graph=True)
        for var, var_loss in ctrl_loss.items():
            var_loss.backward(retain_graph=True)

    def loss(self, y_hat, y, c_hat, c):
        # Dependent variable predictor trained to minimize loss
        dep_loss = self.dep_predictor.loss(y_hat, y)
        # Control variable predictors trained to minimize individual loss
        ctrl_loss = {}
        ctrl_loss_sum = 0
        for var in c.keys():
            ctrl_loss[var] = self.ctrl_predictors[var].loss(c_hat[var], c[var])
            ctrl_loss_sum += ctrl_loss[var]

        # Extractor trained to minimize composite loss
        extractor_loss = dep_loss - 5*ctrl_loss_sum
        return extractor_loss, dep_loss, ctrl_loss

    def score(self, y_hat, y, c_hat, c):
        # Dependent variable predictor trained to minimize loss
        dep_score = self.dep_predictor.score(y_hat, y)
        # Control variable predictors trained to minimize individual loss
        ctrl_score = {}
        for var in c.keys():
            ctrl_score[var] = self.ctrl_predictors[var].score(c_hat[var], c[var])
        return dep_score, ctrl_score



"""
x, y = utils.get_data("EEG_flat", "subject_id")
x = torch.Tensor(x)
y = torch.Tensor(y)
df = pd.read_pickle("../../data/data/numpy_files/df.pkl")

data = Dataset(x, df)
data.dep_variable = "EEG"
data.indep_variable = "math_t1"
data.ctrl_variables += ["subject_id"]
"""

x, y = utils.get_data("digits", "subject_id")
x = torch.Tensor(x)
df = pd.DataFrame(y, columns=["target"])

data = Dataset(x, df)
data.dep_variable = "digits"
data.indep_variable = "target"
data.ctrl_variables += ["target"]


cn = ControlledNetwork()
cn.construct_network(data)
cn.learn(data)
