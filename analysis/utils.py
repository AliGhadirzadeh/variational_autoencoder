import numpy as np

from sklearn.datasets import load_digits
from sklearn.utils import shuffle

from models import *
from sklearn.svm import SVC
from sklearn.svm import SVR

from skopt.space import Real, Integer, Categorical

def get_data(data_string=None, y_variant=None):
    if data_string == None:
        data_string = input("Enter data string (EEG, EEG_flat, digits, ...): ")
    if data_string == "EEG":
        x_path = "../../data/data/snippets.npy"
        x = np.load(x_path)

        if y_variant == None:
            y_variant = input("Enter target variant (scores, ids): ")
        if y_variant == "scores":
            y_path = "../../data/data/scores.npy"
            y = np.load(y_path)
        if y_variant == "ids":
            y_path = "../../data/data/ids.npy"
            y_raw = np.load(y_path)
            y = fix_ids(y_raw)

    if data_string == "EEG_flat":
        x_path = "../../data/data/snippets.npy"
        x = np.load(x_path)
        x = x.reshape(x.shape[0], -1)

        if y_variant == None:
            y_variant = input("Enter target variant (scores, ids): ")
        if y_variant == "scores":
            y_path = "../../data/data/scores.npy"
            y = np.load(y_path)
        if y_variant == "ids":
            y_path = "../../data/data/ids.npy"
            y_raw = np.load(y_path)
            y = fix_ids(y_raw)
        
    if data_string == "digits":
        x, y = load_digits(return_X_y=True)
    elif True:
        pass
    x, y = shuffle(x, y)
    return x, y

def fix_ids(y):
    unique_ids = np.unique(y)
    fixed_y = np.zeros(y.size, dtype=np.int64)
    for idx, id_ in enumerate(unique_ids):
        fixed_y[y == id_] = idx
    return fixed_y

def get_model(model_string=None):
        if model_string == None:
            model_string = input("Enter model string (FFNN_reg, FFNN_clf, Conv1d_clf, TCN_clf, SVR, SVC, ...): ")
        if model_string == "FFNN_reg":
            model = FFNN_reg()
        if model_string == "FFNN_clf":
            model = FFNN_clf()
        if model_string == "Conv1d_reg":
            model = Conv1d_reg()
        if model_string == "Conv1d_clf":
            model = Conv1d_clf()
        if model_string == "TCN_reg":
            model = TemporalConvNet_reg()
        if model_string == "TCN_clf":
            model = TemporalConvNet()
        if model_string == "SVR":
            model = SVR()
        if model_string == "SVC":
            model = SVC()
        elif True:
            pass
        return model


def get_space(estim):
    if type(estim) == SVR or type(estim) == SVC:
        space = [Real(1e-6, 1e+1, prior="log-uniform", name="C"),
                 Real(1e-6, 1e-1, prior="log-uniform", name="gamma"),
                 Integer(1, 8, name="degree"),
                 Categorical(['linear', 'poly', 'rbf'], name="kernel")]
    if type(estim) == FFNN_reg or type(estim) == FFNN_clf:
        space = [Real(1e-6, 1e+1, prior="log-uniform", name="lr"),
                 Integer(1e1, 5e2, prior="log-uniform", name="fc_layer_1"),
                 Integer(1e1, 5e2, prior="log-uniform", name="fc_layer_2"),
                 Integer(1e1, 5e2, prior="log-uniform", name="fc_layer_3"),
                 Integer(1e1, 5e2, prior="log-uniform", name="fc_layer_4")]

    if type(estim) == TemporalConvNet:
        space = [Real(1e-6, 1e+1, prior="log-uniform", name="lr")]
    return space

def plot_learning(estimator, y):
    estimator.history.plot(x="Epoch", y=["Training Error", "Validation Error"])
    estimator_name = str(type(estimator)).partition("'")[2].partition("'")[0].partition(".")[2]
    if "clf" in estimator_name:
        null_score = np.max(np.bincount(y.astype(int))) / y.size
    elif "reg" in estimator_name:
        null_score = np.mean(y)
    else:
        null_score = None

    if null_score:
        plt.hlines(null_score, 0, estimator.history.shape[0], 'r', 'dashed')
        plt.show()
    else:
        plt.show()