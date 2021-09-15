import numpy as np
import matplotlib.pyplot as plt
from matplotlib.pyplot import figure

from misc import *
from sklearn.model_selection import cross_val_score
from skopt.utils import use_named_args
from skopt.callbacks import CheckpointSaver, VerboseCallback
from skopt import gp_minimize
from skopt.plots import plot_evaluations, plot_objective, plot_convergence

# Get data, estimator and hp-space
x, y = get_data()
estim = get_estim()
space = get_space(estim)

# Construct objective
# The objective is the negative mean cv score of the estimator as a function of the hp:s
@use_named_args(space)
def objective(**params):
    estim.set_params(**params)

    return -np.mean(cross_val_score(estim, x, y, cv=5, n_jobs=-1))

# Optimize objective over hp-space
checkpoint_saver = CheckpointSaver("./Checkpoint/checkpoint.pkl")
verbose = VerboseCallback(1)
opt = gp_minimize(objective, space, n_calls=50, callback=[checkpoint_saver, verbose])
print("Best score=%.4f" % opt.fun)


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

# Extract results, save
opt_res = pd.DataFrame(opt.x_iters, columns=opt.space.dimension_names)
opt_res.insert(0, "Score", opt.func_vals)
opt_res.to_pickle("./Results/opt_results.pkl")
opt_res.to_csv("./Results/opt_results.csv")
print("Results saved")
