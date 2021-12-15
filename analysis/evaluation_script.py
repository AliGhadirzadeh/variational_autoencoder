import numpy as np
from utils import *

data_string = "EEG"
test_size = 0.1

model_string = "DG_DGANN"
alpha = 0.01
n_epoch = 50

folds = 30

scores = np.zeros(folds)


for i in range(folds):
	print("Beginning fold " + str(i))
	data = get_data(data_string, binned=False)
	data.subject_wise = True
	data.test_size = test_size

	model = get_model(model_string)
	model.alpha = alpha
	model.n_epoch = n_epoch
	try:
		model.fit(data)
	except:
		model.best_score = np.NaN

	scores[i] = model.best_score
	print("Fold " + str(i) + " ended")
	print("Best score: " + "{:+.3f}".format(scores[i]))
	np.save("scores_", scores)