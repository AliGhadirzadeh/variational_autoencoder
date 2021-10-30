import numpy as np
import pandas as pd
import torch

from model import Dataset
from sklearn.datasets import load_digits

def import_data(data_string):
	if data_string == "EEG":
		x = np.load("../../data/data/numpy_files/snippets.npy")
		df = pd.read_pickle("../../data/data/numpy_files/df.pkl")
		dep_variable = "math_t1"
		ctrl_variable = "subject_id"

		x = torch.from_numpy(x.astype(np.float32))
		y = torch.from_numpy(df[dep_variable].to_numpy())
		c = torch.from_numpy(df[ctrl_variable].to_numpy())
	elif data_string == "digits":
		x_np, y_np = load_digits(return_X_y=True)
		x = torch.Tensor(x_np)
		y = torch.from_numpy(y_np)
		c = torch.from_numpy(y_np)
	data = Dataset(x, y, c)
	return data