# Constructs EEG data and corresponding scores in numpy format given EEG data and scores data
# produced from (eeg_data.zip -> (dawei2zip.sh) -> data.zip -> (unzip) -> data/) and
# (MathTestData.xlsx, TrainingData.xlsx -> (construct_scores.py) -> scores_data.py).
# Directory path contains scores_data.pkl and data/, numpy data will be saved in data/numpy_files

import numpy as np
import pandas as pd
import queue as queue
import argparse
import os
from sklearn.preprocessing import StandardScaler


# take arguments snippet_length and window_length
parser = argparse.ArgumentParser(description='Creates snippets of continous EEG data')
parser.add_argument('--snippet_length', default=500, help='Length of snippets')
parser.add_argument('--window_length', default=500, help='Length of jump between snippet starts (500 equals no overlap between snippets)')
args = parser.parse_args()

# make file structure
directory_path = "../../data/"
data_path = directory_path + "data/"
npy_path = data_path + "numpy_files/"
try:
	os.mkdir(npy_path)
except:
	print("Target directory exists")

def txt2npy(txt_path):
	data = np.loadtxt(txt_path, skiprows=1)
	# Convert time to seconds
	data[:, 0] = data[:, 0] / 1000
	return data

def csv2npy(csv_path):
	data = pd.read_csv(csv_path, sep='\t')
	times = data.values[:, 2]
	times = times.astype(float)
	# Convert time to seconds
	times /= 256
	return times

# Function to get indices of disc times
def get_indices(eeg_time, disc_time):
	disc_indices = np.zeros(disc_time.shape[0], dtype=int)
	for i in range(disc_time.shape[0]):
		disc = disc_time[i]
		# Find two closest elements to disc in eeg_time
		diffs = abs(eeg_time - disc)
		smallest_diffs = np.sort(diffs)[:2]
		# Get their indices
		index_1 = np.where(diffs == smallest_diffs[0])[0]
		index_2 = np.where(diffs == smallest_diffs[1])[0]
		# Get the first of the times
		eeg_ele = min(eeg_time[index_1], eeg_time[index_2])
		# Get index of that time
		disc_index = np.where(eeg_time == eeg_ele)[0]
		# Put in array
		disc_indices[i] = disc_index
	return disc_indices

# Function to get snippets
def get_snippets(sub, eeg_data, disc_time):
	# Convert seconds to indices
	eeg_time = eeg_data[:, 0]
	eeg_data = eeg_data[:, 1:]
	disc_data = get_indices(eeg_time, disc_time)

	# Initialize times, queue and snippet list
	t = 0
	t_max = eeg_data.shape[0]
	disc_queue = queue.Queue()
	if disc_data.shape[0] == 0:
		disc_queue.put(t_max)
	for i in range(disc_data.shape[0]):
		disc_queue.put(disc_data[i])
	t_disc = disc_queue.get()
	snippet_list = []

	# Create snippets
	while t < t_max:
		# If no discontinuity, add snippet
		if t + snippet_length < t_disc:
			snippet_list.append(eeg_data[t:(t+snippet_length), :])
			t += window_length
		# If discontinuity, jump to after discontinuity
		else:
			t = t_disc + 1
			if not disc_queue.empty():
				t_disc = disc_queue.get()
			# End condition
			else:
				if t_disc < t_max:
					t_disc = t_max
				elif t_disc == t_max:
					break
				else:
					print("Error, t_disc > t_max")
					break

	number_of_snippets = len(snippet_list)
	number_of_channels = snippet_list[0].shape[1]
	snippet_array = np.zeros((number_of_snippets, snippet_length, 
							  number_of_channels))
	for snippet_index in range(number_of_snippets):
		snippet_array[snippet_index] = snippet_list[snippet_index]
	np.save(data_path + "snippets/" + sub + "_snippets.npy", snippet_array)

# Initialize parameters and subjects
snippet_length = int(args.snippet_length)
window_length = int(args.window_length)
subs = []
for i in range(1, 52):
	subs.append(str(i))

# Extract snippets
snippet_array = None
for sub in subs:
	print("Processing subject " + sub)
	sub_data_path = data_path + "subs/" + sub + "_transpose"
	try:
		sub_data = txt2npy(sub_data_path)
	except:
		continue
	sub_time_path = data_path + "times/" + sub + ".csv"
	try:
		sub_time = csv2npy(sub_time_path)
	except:
		continue
	sub_snippets = get_snippets(sub, sub_data, sub_time)

# Import scores
df = pd.read_pickle(directory_path + "scores_data.pkl")
key_list = list(df.columns)
list_dict = {}
for key in key_list:
	list_dict[key] = []
data_list = []

for sub in subs:
	sub_path = data_path + "snippets/" +  sub + "_snippets.npy"
	try:
		sub_data = np.load(sub_path)
	except:
		continue
	# Add subjects with complete data ONLY
	if sub_data.shape[2] == 20:
		data_list.append(sub_data)
		ones_template = np.ones(sub_data.shape[0], dtype=float)
		for key in key_list:
			key_value = df.loc[df["raw_subject_id"]==int(sub), key].to_numpy()
			value_array = key_value * ones_template
			list_dict[key].append(value_array)

n_snippets = sum(list(map(lambda sub : sub.shape[0], data_list)))
snippet_length = data_list[0].shape[1]
snippet_channels = 20
snippet_array = np.zeros((n_snippets, snippet_length, snippet_channels))

array_dict = {}
for key in key_list:
	array_dict[key] = np.zeros(n_snippets)


# Add data to numpy array and save
start_index = 0
end_index = 0

# For each subject, extract EEG data from data list and for each key, extract scores
for sub_ind in range(len(data_list)):
	sub_data = data_list[sub_ind]

	start_index = end_index
	end_index += sub_data.shape[0]

	snippet_array[start_index:end_index] = sub_data

	for key in key_list:
		array_dict[key][start_index:end_index] = list_dict[key][sub_ind]

snippet_array = np.swapaxes(snippet_array, 1, 2)

for channel in range(snippet_array.shape[1]):
	scaler = StandardScaler()
	snippet_array[:, channel, :] = scaler.fit_transform(snippet_array[:, channel, :])

np.save(npy_path + "snippets.npy", snippet_array)
for key in key_list:
	np.save(npy_path + key + ".npy", array_dict[key])

y_ids = array_dict["raw_subject_id"]
unique_ids = np.unique(y_ids)
fixed_y = np.zeros(y_ids.size, dtype=np.int64)

for idx, id_ in enumerate(unique_ids):
    fixed_y[y_ids == id_] = idx
array_dict["subject_id"] = fixed_y


df = pd.DataFrame(array_dict)
df.to_pickle(npy_path + "df.pkl")