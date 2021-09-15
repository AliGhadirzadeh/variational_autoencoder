# Test evironment for single estimator debugging
# The test environment-script implements an environment where estimators can be tested on real and test data

import numpy as np
import pandas as pd
from math import sqrt
from sklearn.metrics import mean_squared_error
from estimators import *
from misc import *

# Get estimator and data
x, y = get_data()
estimator = get_estim()
estimator.tqdm_disable = False

# Fit estimator
estimator.fit(x, y)

# Visualize (learning curves)
try:
    estimator.plot_learning()
except:
    print("plot_learning not implemented")



# Save

#save_string = "Results_" + estimator_str + "_" + data_str + "_data"
# make directory
# save figures
# save pd results