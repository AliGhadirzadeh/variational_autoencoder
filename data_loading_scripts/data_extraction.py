# imports
import numpy as np
import pandas as pd
import queue as queue
import argparse

# take arguments snippet_length and window_length
parser = argparse.ArgumentParser(description='Creates snippets to train VAE given EEG data and time discontinuity data')
parser.add_argument('--snippet_length', default=500, help='Length of snippets')
parser.add_argument('--window_length', default=20, help='Length of window of overlap')
args = parser.parse_args()

# make file structure


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
		# If dicontinuity, jump to after discontinuity
		else:
			t = t_disc + 1
			if not disc_queue.empty():
				t_disc = disc_queue.get()
			# End condition
			else:
				if t_disc == t_max:
					break
				else:
					t_disc = t_max

	number_of_snippets = len(snippet_list)
	number_of_channels = snippet_list[0].shape[1]
	snippet_array = np.zeros((number_of_snippets, snippet_length, 
							  number_of_channels))
	for snippet_index in range(number_of_snippets):
		snippet_array[snippet_index] = snippet_list[snippet_index]
	np.save("./data/snippets/" + sub + "_snippets.npy", snippet_array)



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

	sub_data_path = "./data/subs/" + sub + "_transpose"
	try:
		sub_data = txt2npy(sub_data_path)
	except:
		continue

	sub_time_path = "./data/times/" + sub + ".csv"
	print(sub_time_path)
	try:
		sub_time = csv2npy(sub_time_path)
	except:
		continue

	sub_snippets = get_snippets(sub, sub_data, sub_time)

	if snippet_array == None:
		snippet_array = sub_snippets
	else:
		snippet_array.append(sub_snippets)


data_list = []
for sub in subs:
	sub_path = "./data/snippets/" +  sub + "_snippets.npy"
	try:
		sub_data = np.load(sub_path)
	except:
		continue
	# Add subjects with complete data ONLY
	if sub_data.shape[2] == 20:
		data_list.append(sub_data)

# Initiate numpy array
number_of_snippets = sum(list(map(lambda sub : sub.shape[0], data_list)))
snippet_length = data_list[0].shape[1]
snippet_channels = 20
snippet_array = np.zeros((number_of_snippets, snippet_length, snippet_channels))

# Add data to numpy array and save
start_index = 0
end_index = 0
for sub_data in data_list:
	start_index = end_index
	end_index += sub_data.shape[0]
	snippet_array[start_index:end_index] = sub_data

np.save("./data/snippets.npy", snippet_array)