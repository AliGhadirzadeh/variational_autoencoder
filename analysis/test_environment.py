import numpy as np
import pandas as pd
from utils import *
from sklearn.model_selection import train_test_split
from sklearn.metrics import confusion_matrix

# Test evironment for single model debugging
# The test environment-script implements an environment where estimators can be tested on real and test data


# Get estimator and data
data_string = "EEG"
target_string = "latent_t1"
x, y = get_data(data_string, target_string, binned=False)
#x_train, x_test, y_train, y_test = train_test_split(x, y)
y_stratify = get_y_stratify(target_string)
x_train, x_test, y_train, y_test = subject_wise_split(x, y, target_string, stratify=y_stratify)
model = get_model("TCN_reg")
model.tqdm_disable = False
model.subject_wise_split = True
model.target_string = target_string



# Fit estimator
model.fit(x_train, y_train)
print(model.score(x_train, y_train))
print(model.score(x_test, y_test))

# Visualize (learning curves)
plot_learning(model, y)
cm = confusion_matrix(y_test, model.predict(x_test))
plot_confusion_matrix(cm, ["Low", "Mid", "High"])

# Save

#save_string = "Results_" + estimator_str + "_" + data_str + "_data"
# make directory
# save figures
# save pd results
