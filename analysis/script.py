import numpy as np
import pandas as pd
from tqdm import tqdm

from controlled_representation_model import *

import torch
import torch.nn as nn
from torch.utils.tensorboard import SummaryWriter
import torch.optim as optim

from sklearn.base import BaseEstimator
from sklearn.metrics import r2_score, accuracy_score


def import_data(data_string):
	if data_string == "EEG"
		x = np.load("../../data/data/numpy_files/snippets.npy")
		df = pd.read_pickle("../../data/data/numpy_files/df.pkl")
		dep_variable = "math_t1"
		ctrl_variable = "subject_id"

		x = torch.from_numpy(x.astype(np.float32))
		y = torch.from_numpy(df[dep_variable].to_numpy())
		c = torch.from_numpy(df[ctrl_variable].to_numpy())
	elif data_string == "digits":
		#x_np, y_np = 
		x = torch.Tensor(x_np)
		y = torch.from_numpy(y_np)
		c = torch.from_numpy(y_np)
		data = Dataset(x[:500], y[:500], c[:500])
	data = Dataset(x, y, c)
	return data

data_string = "EEG"
data = import_data(data_string)

model = ControlledNetwork()
model.tqdm_disable = False
model.fit(data)


# Digits
"""


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