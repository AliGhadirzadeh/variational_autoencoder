import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

from utils import *
from torch.utils.tensorboard import SummaryWriter

data_string = "EEG"
data = get_data(data_string, binned=False)
data.subject_wise = False
data.test_size = 0.1

train_data, test_data = data.train_test_split()
model_string = "adv_old"
model = get_model(model_string)
model.alpha = 0.5
model.epoch = 50
model.probe_epoch = 200
model.print = True


if input("Log results? (y/n): ") == "y":
	comment = "Model:" + model_string + "_Data:" + data_string + "_alpha:" + str(model.alpha)
	model.writer = SummaryWriter(comment=comment)


#state_dict_path = "./state_dict.pt"
state_dict_path = "./state_dict_adv_model_old.pt"


model.construct_network(data)
model.load_state_dict(torch.load(state_dict_path))
#model.fit(data)

#torch.save(model.state_dict(), state_dict_path)
model.probe(data)