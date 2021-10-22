import numpy as np
import pandas as pd
from utils import *
from sklearn.model_selection import train_test_split

data_string = "EEG"
target_string = "latent_t1"
x, y = get_data(data_string, target_string, binned=True)
y_stratify = get_y_stratify(target_string)

model_list = ["Conv1d_clf"]
n_samples = 30
score_data = np.zeros(n_samples * len(model_list))
model_data = []

for i, model_string in enumerate(model_list):
	print("Sampling " + model_string)
	model_data += [model_string]*n_samples
	for j in range(n_samples):
		print("Sample " + str(j))
		model = get_model(model_string)
		model.tqdm_disable = False
		x_train, x_test, y_train, y_test = subject_wise_split(x, y, stratify=y_stratify)

		model.fit(x_train, y_train)
		score = model.score(x_test, y_test)
		score_data[i*n_samples + j] = score
		print(score)

scores_df = pd.DataFrame({"Score" : score_data, "Model" : model_data})
plot_box(scores_df)
pd.to_pickle(scores_df, "./Results/scores.pkl")