import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

from models.dgdann_20 import DG_DANN_20
from models.probe_y import Probe_Y
from models.probe_d import Probe_D
from models.datasets import *
from torch.utils.tensorboard import SummaryWriter

folder_path = "../../data/rsEEG/data/numpy_files/"

x_np = np.load(folder_path + "snippets.npy")
df = pd.read_pickle(folder_path + "df.pkl")

x = torch.from_numpy(x_np.astype(np.float32))
y = torch.from_numpy(df["math_t1"].to_numpy())
d = torch.from_numpy(df["subject_id"].to_numpy())

data = XYD_Dataset(x, y, d)
data.subject_wise = False
data.test_size = 0.1

model = DG_DANN_20()
model.alpha = 0.3
model.n_epoch = 25
model.print = True


if input("Log results? (y/n): ") == "y":
	comment = "Model:" + "DG_DANN_20" + "__alpha:" + str(model.alpha)
	writer = SummaryWriter(comment=comment)
else:
	writer = None
model.writer = writer

model.fit(data)

# Extract latent representation
model.eval()
z = model.Gf(x).detach()
z = (z - z.mean()) / z.std()

# Validate y
y_data = XY_Dataset(z, y)
y_probe = Probe_Y()
y_probe.writer = writer
y_probe.fit(y_data)
# ~ 0.7-0.9, depending on choice of alpha

# Validate d
d_data = XY_Dataset(z, d)
d_probe = Probe_D()
d_probe.writer = writer
d_probe.fit(d_data)
# ~ 0.2