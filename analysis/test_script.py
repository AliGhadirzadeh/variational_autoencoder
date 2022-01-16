import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

from utils import *
from torch.utils.tensorboard import SummaryWriter

data_string = "XYD"
data = get_data(data_string, binned=False)
data.subject_wise = False
data.test_size = 0.1

train_data, test_data = data.train_test_split()
model_string = "dgdann_20"
model = get_model(model_string)
model.alpha = 0.5
model.n_epoch = 22
model.print = True
#model.save_path = "./pretrained_dg_dann.pt"


if input("Log results? (y/n): ") == "y":
	comment = "Model:" + model_string + "_Data:" + data_string + "_alpha:" + str(model.alpha)
	model.writer = SummaryWriter(comment=comment)

model.fit(data)
model.probe(data)
