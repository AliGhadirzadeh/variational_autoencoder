import numpy as np
import pandas as pd
import torch

from models.eegnet import EEGNet
from models.eegtcn_8 import EEGTCN_8
from models.eegtcn_20 import EEGTCN_20
from models.dgdann_8 import DG_DANN_8
from models.dgdann_20 import DG_DANN_20
from models.datasets import *

def get_data(data_string, binned=False, n_levels=3):
	folder_path = "../../data/rsEEG/data/numpy_files/"

	if data_string == "XY":
		x = np.load(folder_path + "snippets.npy")
		df = pd.read_pickle(folder_path + "df.pkl")
		dep_variable = "math_d"
		x = torch.from_numpy(x.astype(np.float32))
		y = torch.from_numpy(df[dep_variable].to_numpy())
		data = XY_Dataset(x, y, binned=binned, n_levels=n_levels) if binned else XY_Dataset(x, y)

	if data_string == "XYD":
		x = np.load(folder_path + "snippets.npy")
		df = pd.read_pickle(folder_path + "df.pkl")
		dep_variable = "math_d"
		x = torch.from_numpy(x.astype(np.float32))
		y = torch.from_numpy(df[dep_variable].to_numpy())
		d = torch.from_numpy(df["subject_id"].to_numpy())
		data = XYD_Dataset(x, y, d, binned=binned, n_levels=n_levels) if binned else XYD_Dataset(x, y, d)
		
	return data
	

def get_model(model_string):

	if model_string == "eegnet":
		return EEGNET()

	elif model_string == "eegtcn_8":
		return EEGTCN_8()

	elif model_string == "eegtcn_20":
		return EEGTCN_20()

	elif model_string == "dgdann_8":
		return DG_DANN_8()

	elif model_string == "dgdann_20":
		return DG_DANN_20()