import numpy as np
import pandas as pd
from tqdm import tqdm 

import utils
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
        self.conv = True

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


class ControlledNetwork(nn.Module, BaseEstimator):
    """docstring for CompositeNetwork"""
    def __init__(self, lr=3e-4, writer=None):
        super(ControlledNetwork, self).__init__()
        self.lr = lr

        self.extractor = ExtractorNetwork()
        self.dep_predictor = PredictorNetwork()
        self.ctrl_predictor = PredictorNetwork()

        self.n_epoch = 100
        
        self.tqdm_disable = True
        self.writer = writer

    def construct_network(self, data):
        with torch.no_grad():
            x, y, c = data[:]
            self.extractor.construct_network(x)
            x_latent = self.extractor(x)
            self.dep_predictor.construct_network(x_latent, y)
            self.ctrl_predictor.construct_network(x_latent, c)
            if self.writer:
                self.writer.add_graph(self, x[0])
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
                self.evaluate(val_data, epoch)

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
    
    def evaluate(self, data, epoch):
        x, y, c = data[:]
        y_hat, c_hat = self.forward(x)
        extractor_loss = self.extractor_loss(y_hat, y, c_hat, c)
        dep_loss = self.dep_predictor.loss(y_hat, y)
        ctrl_loss = self.ctrl_predictor.loss(c_hat, c)
        dep_score = self.dep_predictor.score(y_hat, y)
        ctrl_score = self.ctrl_predictor.score(c_hat, c)

        if self.writer:
            self.writer.add_scalars("Loss", {"extractor" : extractor_loss,
                                             "dep_predictor" : dep_loss,
                                             "ctrl_predictor" : ctrl_loss}, 
                                             epoch)
            self.writer.add_scalar('Accuracy/dep_predictor', dep_score, epoch)
            self.writer.add_scalar('Accuracy/ctrl_predictor', ctrl_score, epoch)


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
        

# EEG

x, y = utils.get_data("EEG", "subject_id")

df = pd.read_pickle("../../data/data/numpy_files/df.pkl")
dep_variable = "math_t1"
ctrl_variable = "subject_id"

x = torch.from_numpy(x.astype(np.float32))
y = torch.from_numpy(df[dep_variable].to_numpy())
c = torch.from_numpy(df[ctrl_variable].to_numpy())

print(x.dtype)

data = Dataset(x, y, c)
model = ControlledNetwork()
model.tqdm_disable = False
#model.fit(data)


# Digits
"""
x_np, y_np = utils.get_data("digits", "subject_id")
x = torch.Tensor(x_np)
y = torch.from_numpy(y_np)
c = torch.from_numpy(y_np)
data = Dataset(x[:500], y[:500], c[:500])

from sklearn.model_selection import cross_val_score
from skopt.space import Real, Integer, Categorical
from skopt.utils import use_named_args
from skopt.callbacks import CheckpointSaver, VerboseCallback
from skopt import gp_minimize
from skopt.plots import plot_evaluations, plot_objective, plot_convergence

space = [Real(5e-5, 1e-2, prior="log-uniform", name="lr")]
model = ControlledNetwork()
model.tqdm_disable = False

# Construct objective
# The objective is the negative mean cv score of the estimator as a function of the hp:s
@use_named_args(space)
def objective(**params):
    model.set_params(**params)
    train_data, test_data = data.train_test_split()
    model.construct_network(data)
    print(model.score(test_data))
    model.fit(train_data)
    return -model.score(test_data)

# Optimize objective over hp-space
#checkpoint_saver = CheckpointSaver("./Checkpoint/checkpoint.pkl")
verbose = VerboseCallback(1)
opt = gp_minimize(objective, space, n_calls=50, callback=[verbose])
print("Best score=%.4f" % opt.fun)

import matplotlib.pyplot as plt

# Plot convergence trace, evaluations and objective function
plot_convergence(opt)
plt.savefig("./Results/conv_plot.png")
plt.show()

plt.rcParams["font.size"] = 7
#plt.rcParams["figure.autolayout"] = True
_ = plot_evaluations(opt, bins=10)
plt.savefig("./Results/eval_plot.png")
plt.show()

plt.rcParams["font.size"] = 7
#plt.rcParams["figure.autolayout"] = True
_ = plot_objective(opt)
plt.savefig("./Results/obj_plot.png")
plt.show()
"""