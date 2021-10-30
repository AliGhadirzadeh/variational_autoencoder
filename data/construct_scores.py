# Construct cognitive scores from test data
# Run script in directory containing MathTestData.xlsx and TrainingData.xlsx
# Produces a pickled Pandas file scores_data.pkl

import numpy as np
import pandas as pd
from sklearn.decomposition import FactorAnalysis
import matplotlib.pyplot as plt


file_path = "../../data/"

math_scores = pd.read_excel(file_path + "MathTestData.xlsx")

xls = pd.ExcelFile(file_path + "TrainingData.xlsx")
nl_scores = pd.read_excel(xls, "numberline")
wm_scores = pd.read_excel(xls, "working memory")
rot_scores = pd.read_excel(xls, "rotation")

subject_id = math_scores.loc[:, "Participant Code"].to_numpy()
subject_id = np.expand_dims(subject_id, axis=1)


plot = False

# Get math scores
math_t1 = math_scores.loc[:, "PreTest"].to_numpy()
math_t1 = np.expand_dims(math_t1, axis=1)
if plot:
	_ = plt.hist(math_t1, bins="auto")
	plt.title("math_t1")
	plt.show()

math_t2 = math_scores.loc[:, "PostTest"].to_numpy()
math_t2 = np.expand_dims(math_t2, axis=1)
if plot:
	_ = plt.hist(math_t2, bins="auto")
	plt.title("math_t2")
	plt.show()

math_d = math_t2 - math_t1


# Get numberline scores
nl_t1 = nl_scores.loc[:, "1stWeek"].to_numpy()
nl_t1 = np.expand_dims(nl_t1, axis=1)
if plot:
	_ = plt.hist(nl_t1, bins="auto")
	plt.title("nl_t1")
	plt.show()

nl_t2 = nl_scores.loc[:, "5thWeek"].to_numpy()
nl_t2 = np.expand_dims(nl_t2, axis=1)
if plot:
	_ = plt.hist(nl_t2, bins=20)
	plt.title("nl_t2")
	plt.show()

nl_d = nl_t2 - nl_t1


# Get working memory scores
wm_t1 = wm_scores.loc[:, "1stWeek"].to_numpy()
wm_t1 = np.expand_dims(wm_t1, axis=1)
if plot:
	_ = plt.hist(wm_t1, bins="auto")
	plt.title("wm_t1")
	plt.show()

wm_t2 = wm_scores.loc[:, "5thWeek"].to_numpy()
wm_t2 = np.expand_dims(wm_t2, axis=1)
if plot:
	_ = plt.hist(wm_t2, bins="auto")
	plt.title("wm_t2")
	plt.show()

wm_d = wm_t2 - wm_t1


# Get rotation scores
rot_t1 = rot_scores.loc[:, "1stWeek"].to_numpy()
rot_t1 = np.expand_dims(rot_t1, axis=1)
if plot:
	_ = plt.hist(rot_t1, bins="auto")
	plt.title("rot_t1")
	plt.show()

rot_t2 = rot_scores.loc[:, "5thWeek"].to_numpy()
rot_t2 = np.expand_dims(rot_t2, axis=1)
if plot:
	_ = plt.hist(rot_t2, bins="auto")
	plt.title("rot_t2")
	plt.show()

rot_d = rot_t2 - rot_t1


# Calculate latent scores
scores_t1 = np.concatenate([math_t1, nl_t1, wm_t1, rot_t1], axis=1)
scores_t1 = np.nan_to_num(scores_t1)

scores_t2 = np.concatenate([math_t2, nl_t2, wm_t2, rot_t2], axis=1)
scores_t2 = np.nan_to_num(scores_t2)

transformer_t1 = FactorAnalysis(n_components=1)
latent_t1 = transformer_t1.fit_transform(scores_t1)
if plot:
	_ = plt.hist(latent_t1, bins="auto")
	plt.title("latent_t1")
	plt.show()

transformer_t2 = FactorAnalysis(n_components=1)
latent_t2 = transformer_t2.fit_transform(scores_t2)
if plot:
	_ = plt.hist(latent_t2, bins="auto")
	plt.title("latent_t2")
	plt.show()

latent_d = latent_t2 - latent_t1


# Construct DataFrame
labels = ["raw_subject_id", 
		  "math_t1", "math_t2", "math_d", 
		  "nl_t1", "nl_t2", "nl_d",
		  "wm_t1", "wm_t2", "wm_d", 
		  "rot_t1", "rot_t2", "rot_d",
		  "latent_t1", "latent_t2", "latent_d"]

data_list = [subject_id, 
			 math_t1, math_t2, math_d,
			 nl_t1, nl_t2, nl_d,
			 wm_t1, wm_t2, wm_d,
			 rot_t1, rot_t2, rot_d,
			 latent_t1, latent_t2, latent_d]
data = np.concatenate(data_list, axis=1)

df = pd.DataFrame(data, columns=labels)
df.to_pickle(file_path + "scores_data.pkl")