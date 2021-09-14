# Estimator objects
import copy
from math import sqrt
import numpy as np
import pandas as pd
from tqdm import tqdm 
import matplotlib.pyplot as plt

from sklearn.metrics import mean_squared_error, log_loss, r2_score
from sklearn.base import BaseEstimator
from sklearn.svm import SVR

import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torch.utils.data import TensorDataset, DataLoader

from skopt import BayesSearchCV
from skopt.plots import plot_objective, plot_histogram, plot_convergence
from skopt import callbacks
from skopt.callbacks import CheckpointSaver

class ConstructData(object):

    def __init__(self, data_string="prompt"):
        super(ConstructData, self).__init__()
        self.data_string = data_string

    def get_data(self):
        if self.data_string == "prompt":
            self.data_string = input("Enter data string (EEG, test, test_SVR): ")
        if self.data_string == "test":
            x = np.random.normal(0, 1, [10000, 500, 20])
            x_1 = np.sum(x, axis=1)
            y = np.sum(x_1, axis=1)
        if self.data_string == "test_SVR":
            x = np.random.normal(0, 1, [10000, 20])
            y = np.sum(x, axis=1)
        if self.data_string == "EEG":
            snippet_path = "../data/data/snippets.npy" 
            scores_path = "../data/data/scores.npy"
            x = np.load(snippet_path)
            y = np.load(scores_path)
        elif True:
            pass
        return x, y


class ConstructEstim(object):

    def __init__(self, estimator_string="prompt"):
        super(ConstructEstim, self).__init__()
        self.estimator_string = estimator_string

    def get_estim(self):
        if self.estimator_string == "prompt":
            self.prompt()
        if self.estimator_string == "FFNN":
            estimator = FFNN()
        if self.estimator_string == "SVR":
            estimator = SVR()
        elif True:
            pass
        return estimator

    def prompt(self):
        self.estimator_string = input("Enter estimator string (FFNN, SVR, ...): ")
        return


class ConstructOpt(object):

    def __init__(self):
        estim_constructor = ConstructEstim()
        estimator = estim_constructor.get_estim()

        space = self.get_space()
        n_iter = 10
        cv = 2
        self.opt = BayesSearchCV(
                                    estimator,
                                    space,
                                    n_iter=n_iter,
                                    cv=cv
                                )

    def get_space(self):
        space_string = input("Enter space string (FFNN, SVR, ...): ")
        if space_string == "SVR":
            space = {
                        'C': (1e-6, 1e+3, 'log-uniform'),
                        'gamma': (1e-6, 1e0, 'log-uniform'),
                        'degree': (1, 8),  # integer valued parameter
                        'kernel': ['linear', 'poly', 'rbf'],  # categorical parameter
                    }
        return space

    def get_opt(self):
        return self.opt


class FFNN(BaseEstimator, nn.Module):

    def __init__(self, lr=3e-2, layer1=100, layer2=100, layer3=100, layer4=100, layer5=100):
        super(FFNN, self).__init__()
        self.lr = lr
        self.layer1 = layer1
        self.layer2 = layer2
        self.layer3 = layer3
        self.layer4 = layer4
        self.layer5 = layer5

        self.n_epoch = 100
        self.val_frac = 0.2
        self.patience = 100
        cols = ["Epoch", "Training error", "Validation error"]
        array = np.empty((self.n_epoch, len(cols)))
        array[:] = np.nan
        self.history = pd.DataFrame(array, columns=cols)
        self.x = None
        self.y = None
        self.tqdm_disable = True

        self.fc1 = nn.Linear(10000, self.layer1)
        self.fc2 = nn.Linear(self.layer1, self.layer2)
        self.fc3 = nn.Linear(self.layer2, self.layer3)
        self.fc4 = nn.Linear(self.layer3, self.layer4)
        self.fc5 = nn.Linear(self.layer4, self.layer5)
        self.fc5 = nn.Linear(self.layer5, 1)
        
    def convert_data(self, x, y):
        x = np.reshape(x, (x.shape[0], x.shape[1]*x.shape[2]))
        y = np.reshape(y, (-1, 1))
        x = torch.Tensor(x)
        y = torch.Tensor(y)
        return x, y


    def fit(self, x, y):
        self.x = copy.deepcopy(x)
        self.y = copy.deepcopy(y)
        n_val = int(x.shape[0] * self.val_frac)
        x_train = x[n_val:]
        y_train = y[n_val:]
        x_val = x[:n_val]
        y_val = y[:n_val]
        
        # Convert to Pytorch Tensors
        x_train, y_train = self.convert_data(x_train, y_train)
        x_val, y_val = self.convert_data(x_val, y_val)

        # Initialize training/validation loop
        val_counter = 0
        best_score = -np.inf
        best_model = None

        opt = optim.Adam(self.parameters(), self.lr)
        for epoch in tqdm(range(self.n_epoch), disable=self.tqdm_disable):
            opt.zero_grad()
            y_hat = self.predict(x_train)
            loss = F.mse_loss(y_hat, y_train)
            loss.backward()
            opt.step()

            train_score = self.score(x_train, y_train)
            val_score = self.score(x_val, y_val)
            self.history.loc[epoch, "Epoch"] = epoch
            self.history.loc[epoch, "Training Error"] = train_score
            self.history.loc[epoch, "Validation Error"] = val_score

            if val_score > best_score:
                best_model = copy.deepcopy(self.state_dict())
                best_score = val_score
                val_counter = 0
            else:
                val_counter += 1
                if val_counter == self.patience:
                    self.load_state_dict(best_model)
                    self.history.dropna(how="all")
                    break
        self.history.dropna(how="all")
        return

    def score(self, x, y):
        if isinstance(x, np.ndarray) and isinstance(y, np.ndarray):
            x, y = self.convert_data(x, y)
        y_hat = self.predict(x).detach().numpy()
        score = r2_score(y, y_hat)
        #error = sqrt(mean_squared_error(y, y_hat))
        return score

    def predict(self, x):
        x = F.relu(self.fc1(x))
        x = F.relu(self.fc2(x))
        x = F.relu(self.fc3(x))
        x = F.relu(self.fc4(x))
        x = self.fc5(x)
        return x

    def plot_learning(self):
        self.history.plot(x="Epoch", y=["Training Error", "Validation Error"])
        y_mean = np.mean(self.y) * np.ones(self.y.shape[0])
        null_score = 0
        #null_error = sqrt(mean_squared_error(self.y, y_mean))
        plt.hlines(null_score, 0, self.n_epoch, 'r', 'dashed')
        plt.show()
