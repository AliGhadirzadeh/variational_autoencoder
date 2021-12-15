import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

from utils import *
from models.datasets import *
from models.eegtcn_8 import EEGTCN_8

folder_path = "../../data/SSVEP/data/numpy_files/"

x = np.load(folder_path + "eeg.npy")
x = torch.from_numpy(x.astype(np.float32))
y = np.load(folder_path + "ids.npy")
y = torch.from_numpy(y.astype(np.int))

data = XY_Dataset(x, y)
x, y = data[:]
data.test_size = 0.1

model = EEGTCN_8()
model.print = True
model.n_epoch = 300

model.fit(data)