import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

from utils import *
from torch.utils.tensorboard import SummaryWriter

# "EEG", "EEG_ctrl", "digits", "digits_ctrl" 
data_string = "EEG"
data = get_data(data_string, binned=True)

# "ctrl", "conv1d"
model_string = "eegtcn"
model = get_model(model_string)
comment = "Model:" + model_string + "_Data:" + data_string

model.writer = SummaryWriter(comment=comment)
model.construct_network(data)
model.fit(data)