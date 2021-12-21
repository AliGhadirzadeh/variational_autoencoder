import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

from models.dgdann_20 import DG_DANN_20
from models.datasets import *

# change paths to relevant files
folder_path = "../../data/rsEEG/data/numpy_files/"

x = np.load(folder_path + "snippets.npy")
df = pd.read_pickle(folder_path + "df.pkl")

x = torch.from_numpy(x.astype(np.float32))
y = torch.from_numpy(df["subject_id"].to_numpy())

data = XY_Dataset(x, y)

model = DG_DANN_20()
model.load_state_dict(torch.load("./pretrained_dg_dann.pt"))
model.eval()

x, y = data[:]

features = model.Gf(x)

