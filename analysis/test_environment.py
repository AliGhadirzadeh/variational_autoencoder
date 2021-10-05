# Test evironment for single model debugging
# The test environment-script implements an environment where estimators can be tested on real and test data

import numpy as np
import pandas as pd
from utils import *

# Get estimator and data
x, y = get_data("EEG", "scores")
model = get_model("Conv1d_reg")
model.tqdm_disable = False


# Fit estimator
model.fit(x, y)


# Visualize (learning curves)
plot_learning(model, y)

# Save

#save_string = "Results_" + estimator_str + "_" + data_str + "_data"
# make directory
# save figures
# save pd results
