import numpy as np
import pandas as pd
import torch

from models.conv1d_model import Conv1dNetwork
from models.controlled_model import ControlledNetwork
from models.test_model import TestNetwork
from models.eegnet_model import EEGNetwork
from models.eegtcn_model import EEGTCN
from models.datasets import *
from sklearn.datasets import load_digits

def get_data(data_string, binned=False):
	if data_string == "EEG":
		x = np.load("../../data/data/numpy_files/snippets.npy")
		df = pd.read_pickle("../../data/data/numpy_files/df.pkl")
		dep_variable = "math_t1"
		x = torch.from_numpy(x.astype(np.float32))
		y = torch.from_numpy(df[dep_variable].to_numpy())
		data = Dataset(x, y, binned=binned) if binned else Dataset(x, y)

	elif data_string == "EEG_ctrl":
		x = np.load("../../data/data/numpy_files/snippets.npy")
		df = pd.read_pickle("../../data/data/numpy_files/df.pkl")
		dep_variable = "math_t1"
		ctrl_variable = "subject_id"
		x = torch.from_numpy(x.astype(np.float32))
		y = torch.from_numpy(df[dep_variable].to_numpy())
		c = torch.from_numpy(df[ctrl_variable].to_numpy())
		data = CtrlDataset(x, y, c, binned=binned) if binned else CtrlDataset(x, y, c)

	elif data_string == "digits":
		x_np, y_np = load_digits(return_X_y=True)
		x = torch.Tensor(x_np)
		y = torch.from_numpy(y_np)
		data = Dataset(x, y)

	elif data_string == "digits_ctrl":
		x_np, y_np = load_digits(return_X_y=True)
		x = torch.Tensor(x_np)
		y = torch.from_numpy(y_np)
		c = torch.from_numpy(y_np)
		data = CtrlDataset(x, y, c)
	return data

def get_model(model_string):
	if model_string == "conv1d":
		return Conv1dNetwork()

	elif model_string == "ctrl":
		return ControlledNetwork()

	elif model_string == "test":
		return TestNetwork()

	elif model_string == "eegnet":
		return EEGNetwork()

	elif model_string == "eegtcn":
		return EEGTCN()
