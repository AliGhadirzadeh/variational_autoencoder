import numpy as np

from sklearn.datasets import load_digits

from estimators import *
from sklearn.svm import SVC
from sklearn.svm import SVR

from skopt.space import Real, Integer, Categorical

def get_data():
    data_string = input("Enter data string (EEG, EEG_flat, digits, ...): ")
    if data_string == "test":
        x = np.random.normal(0, 1, [10000, 500, 20])
        x_1 = np.sum(x, axis=1)
        y = np.sum(x_1, axis=1)
    if data_string == "test_SVR":
        x = np.random.normal(0, 1, [10000, 20])
        y = np.sum(x, axis=1)
    if data_string == "EEG":
        snippet_path = "../../data/data/snippets.npy" 
        scores_path = "../../data/data/scores.npy"
        x = np.load(snippet_path)
        y = np.load(scores_path)
    if data_string == "EEG_flat":
        snippet_path = "../../data/data/snippets.npy" 
        scores_path = "../../data/data/scores.npy"
        x = np.load(snippet_path)
        x = x.reshape(x.shape[0], -1)
        print(x.shape)
        y = np.load(scores_path)
    if data_string == "digits":
        x, y = load_digits(return_X_y=True)
    elif True:
        pass
    return x, y


def get_estim():
        estimator_string = input("Enter estimator string (FFNN_reg, FFNN_clf, SVR, SVC, ...): ")
        if estimator_string == "FFNN_reg":
            estimator = FFNN_reg()
        if estimator_string == "FFNN_clf":
            estimator = FFNN_clf()
        if estimator_string == "SVR":
            estimator = SVR()
        if estimator_string == "SVC":
            estimator = SVC()
        elif True:
            pass
        return estimator


def get_space(estim):
    if type(estim) == SVR or type(estim) == SVC:
        space = [Real(1e-6, 1e+1, prior="log-uniform", name="C"),
                 Real(1e-6, 1e-1, prior="log-uniform", name="gamma"),
                 Integer(1, 8, name="degree"),
                 Categorical(['linear', 'poly', 'rbf'], name="kernel")]
    if type(estim) == FFNN_reg or type(estim) == FFNN_clf:
        space = [Real(1e-6, 1e+1, prior="log-uniform", name="lr"),
                 Integer(1e1, 5e2, prior="log-uniform", name="layer1"),
                 Integer(1e1, 5e2, prior="log-uniform", name="layer2"),
                 Integer(1e1, 5e2, prior="log-uniform", name="layer3"),
                 Integer(1e1, 5e2, prior="log-uniform", name="layer4")]
    return space