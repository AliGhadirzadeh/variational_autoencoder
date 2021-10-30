import numpy as np
import pandas as pd
import torch

from model import *
from utils import *
from torch.utils.tensorboard import SummaryWriter

from skopt.space import Real, Integer, Categorical
from skopt.utils import use_named_args
from skopt.callbacks import CheckpointSaver, VerboseCallback
from skopt import gp_minimize


data = import_data("digits")
space = [Real(5e-5, 1e-2, prior="log-uniform", name="lr")]
model = ControlledNetwork()

# Construct objective
# Add cross-validation and GPU support
@use_named_args(space)
def objective(**params):
    model.set_params(**params)
    train_data, test_data = data.train_test_split()
    model.construct_network(data)
    model.fit(train_data)
    return -model.score(test_data)

# Optimize objective over hp-space
#checkpoint_saver = CheckpointSaver("./Checkpoint/checkpoint.pkl")
verbose = VerboseCallback(1)
opt = gp_minimize(objective, space, n_calls=50, callback=[verbose])


import matplotlib.pyplot as plt
from skopt.plots import plot_evaluations, plot_objective, plot_convergence

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