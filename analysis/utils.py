import itertools
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns

from sklearn.datasets import load_digits
from sklearn.utils import shuffle
from sklearn.model_selection import train_test_split

from models import *
from sklearn.svm import SVC
from sklearn.svm import SVR

from skopt.space import Real, Integer, Categorical

numpy_path = "../../data/data/numpy_files/"

def get_data(data_string=None, y_variant=None, binned=True):
    if data_string == None:
        data_string = input("Enter data string (EEG, EEG_flat, digits, ...): ")
    if data_string == "EEG":
        x_path = numpy_path + "snippets.npy"
        x = np.load(x_path)
    elif data_string == "EEG_flat":
        x_path = numpy_path + "snippets.npy"
        x = np.load(x_path)
        x = x.reshape(x.shape[0], -1)

    if y_variant == None:
        y_variant = input("Enter target variant (subject_id or measure [latent, math, rot, nl, wm]) + _ + timepoint [t1, t2, d] (i.e. 'math_t2: ")
    y_df = pd.read_pickle(numpy_path + "df.pkl")
    y = y_df[y_variant].to_numpy()
    if y_variant == "subject_id":
        y = fix_ids(y)
    else:
        if binned:
            y = bin_scores(y)
    if data_string == "digits":
        x, y = load_digits(return_X_y=True)
    elif True:
        pass
    return x, y


def fix_ids(y):
    unique_ids = np.unique(y)
    fixed_y = np.zeros(y.size, dtype=np.int64)
    for idx, id_ in enumerate(unique_ids):
        fixed_y[y == id_] = idx
    return fixed_y


def bin_scores(y, n_levels=3):
    binned_y = np.zeros(y.size, dtype=np.int64)
    unique_y = np.unique(y)
    n_ids = int(unique_y.size / n_levels)
    for i in range(n_levels):
        start_index = n_ids * i
        end_index = -1 if i == n_levels-1 else n_ids * (i+1)
        level_ids = unique_y[n_ids*i:n_ids*(i+1)]
        binned_y[np.isin(y, level_ids)] = i
    return binned_y


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
            model = TemporalConvNet_clf()
        if model_string == "SVR":
            model = SVR()
        if model_string == "SVC":
            model = SVC()
        elif True:
            pass
        return model


def get_space(model):
    if type(model) == SVR or type(model) == SVC:
        space = [Real(1e-6, 1e+1, prior="log-uniform", name="C"),
                 Real(1e-6, 1e-1, prior="log-uniform", name="gamma"),
                 Integer(1, 8, name="degree"),
                 Categorical(['linear', 'poly', 'rbf'], name="kernel")]
    if type(model) == FFNN_reg or type(model) == FFNN_clf:
        space = [Real(1e-6, 1e+1, prior="log-uniform", name="lr"),
                 Integer(1e1, 5e2, prior="log-uniform", name="fc_layer_1"),
                 Integer(1e1, 5e2, prior="log-uniform", name="fc_layer_2"),
                 Integer(1e1, 5e2, prior="log-uniform", name="fc_layer_3"),
                 Integer(1e1, 5e2, prior="log-uniform", name="fc_layer_4")]

    if type(model) == TemporalConvNet_clf or type(model) == TemporalConvNet_reg:
        space = [Real(1e-6, 1e+1, prior="log-uniform", name="lr")]
    return space


def plot_learning(model, y):
    model.history.plot(x="Epoch", y=["Training Error", "Validation Error"])
    model_name = str(type(model)).partition("'")[2].partition("'")[0].partition(".")[2]
    if "clf" in model_name:
        null_score = np.max(np.bincount(y.astype(int))) / y.size
    elif "reg" in model_name:
        null_score = np.mean(y)
    else:
        null_score = None

    if null_score:
        plt.hlines(null_score, 0, model.history.shape[0], 'r', 'dashed')
        plt.show()
    else:
        plt.show()


def get_y_stratify(target_string):
    y_df = pd.read_pickle(numpy_path + "df.pkl")
    ids = y_df["subject_id"].to_numpy()
    scores_list = []
    y = y_df[target_string].to_numpy()
    binned_y = bin_scores(y, n_levels=3)
    for i in np.unique(ids):
        scores_list.append(binned_y[ids==i][0])
    y_stratify = np.array(scores_list)
    return y_stratify

"""
def subject_wise_split(x, y, test_size=0.2, stratify=None):
    y_df = pd.read_pickle(numpy_path + "df.pkl")
    ids = y_df["subject_id"].to_numpy()
    unique_ids = np.unique(ids)
    train_ids, test_ids = train_test_split(unique_ids, 
                                           test_size=test_size,
                                           shuffle=True,
                                           stratify=stratify)
    train_indices = np.where(np.in1d(ids, train_ids))
    test_indices = np.where(np.in1d(ids, test_ids))
    x_train = x[train_indices]
    x_test = x[test_indices]
    y_train = y[train_indices[0]]
    y_test = y[test_indices[0]]
    return x_train, x_test, y_train, y_test
"""


def subject_wise_split(x, y, target_string, test_size=0.2, stratify=None):
    ids = get_ids(y, target_string)
    y_df = pd.read_pickle(numpy_path + "df.pkl")
    ids = y_df["subject_id"].to_numpy()
    unique_ids = np.unique(ids)
    train_ids, test_ids = train_test_split(unique_ids, 
                                           test_size=test_size,
                                           shuffle=True,
                                           stratify=stratify)
    train_indices = np.where(np.in1d(ids, train_ids))
    test_indices = np.where(np.in1d(ids, test_ids))
    x_train = x[train_indices]
    x_test = x[test_indices]
    y_train = y[train_indices[0]]
    y_test = y[test_indices[0]]
    return x_train, x_test, y_train, y_test

def get_ids(y, target_string):
    y_df = pd.read_pickle(numpy_path + "df.pkl")
    y_ids = np.zeros(y.shape[0])
    print(np.unique(y))
    for score in np.unique(y):
        print(score)
        print(np.unique(y_df[target_string].to_numpy()))
        print(y_df[target_string]==1.1447)
        subject_df = y_df.loc[y_df[target_string]==score]
        subject_id = subject_df["subject_id"].iloc[0]
        y_ids[y==score] = subject_id
    return y_ids

def plot_confusion_matrix(cm,
                          target_names,
                          title='Confusion matrix',
                          cmap=None,
                          normalize=True):
    
    accuracy = np.trace(cm) / np.sum(cm).astype('float')
    misclass = 1 - accuracy

    if cmap is None:
        cmap = plt.get_cmap('Blues')

    plt.figure(figsize=(8, 6))
    plt.imshow(cm, interpolation='nearest', cmap=cmap)
    plt.title(title)
    plt.colorbar()

    if target_names is not None:
        tick_marks = np.arange(len(target_names))
        plt.xticks(tick_marks, target_names, rotation=45)
        plt.yticks(tick_marks, target_names)

    if normalize:
        cm = cm.astype('float') / cm.sum(axis=1)[:, np.newaxis]


    thresh = cm.max() / 1.5 if normalize else cm.max() / 2
    for i, j in itertools.product(range(cm.shape[0]), range(cm.shape[1])):
        if normalize:
            plt.text(j, i, "{:0.4f}".format(cm[i, j]),
                     horizontalalignment="center",
                     color="white" if cm[i, j] > thresh else "black")
        else:
            plt.text(j, i, "{:,}".format(cm[i, j]),
                     horizontalalignment="center",
                     color="white" if cm[i, j] > thresh else "black")

    plt.tight_layout()
    plt.ylabel('True label')
    plt.xlabel('Predicted label\naccuracy={:0.4f}; misclass={:0.4f}'.format(accuracy, misclass))
    plt.show()


def plot_box(df):
    ax = sns.boxplot(x="Model", y="Score", data=df)
    ax = sns.swarmplot(x="Model", y="Score", data=df, color=".25")
    plt.show()