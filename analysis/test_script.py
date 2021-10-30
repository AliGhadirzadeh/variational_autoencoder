import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

from model import *
from utils import *
from torch.utils.tensorboard import SummaryWriter

data_string = "digits"
data = import_data(data_string)
writer = SummaryWriter()

model = ControlledNetwork(writer=writer)
#writer.add_graph(model, data)
model.fit(data)