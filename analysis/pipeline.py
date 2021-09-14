# The pipeline-script implements a preprecsessing/hp-search/evaluation-pipeline

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

from estimators import *

from skopt import BayesSearchCV
from skopt.plots import plot_evaluations, plot_objective, plot_convergence
from skopt import callbacks
from skopt.callbacks import CheckpointSaver

data_constructor = ConstructData()
x, y = data_constructor.get_data()

opt_constructor = ConstructOpt()
opt = opt_constructor.get_opt()

checkpoint_saver = CheckpointSaver("./Checkpoint/checkpoint.pkl")
opt.fit(x, y, callback=[checkpoint_saver])

_ = plot_objective(opt.optimizer_results_[0],
                   dimensions=["C", "gamma", "degree", "kernel"])
_ = plot_evaluations(opt.optimizer_results_[0],
                   dimensions=["C", "gamma", "degree", "kernel"])
plt.show()

cv_results = pd.DataFrame.from_dict(opt.cv_results_)
cv_results.to_pickle("./Results/cv_results.pkl")
cv_results.to_csv("./Results/cv_results.csv")

print("Results saved")