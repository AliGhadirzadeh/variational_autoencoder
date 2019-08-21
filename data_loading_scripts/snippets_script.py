import numpy as np
import queue as queue

import argparse

parser = argparse.ArgumentParser(description='Creates snippets to train VAE given EEG data and time discontinuity data')
parser.add_argument('--snippet_length', default=500, help='Length of snippets')
parser.add_argument('--window_length', default=100, help='Length of window of overlap')
args = parser.parse_args()

# Initialize parameters and subjects
snippet_length = int(args.snippet_length)
window_length = int(args.window_length)
subject_list = []
for i in range(1, 52):
	subject_list.append(str(i))

for subject in subject_list:
	print(subject)
	# Load data, create queue
	# eeg_data is numpy array containing channel voltages over time
	# disc_data is a numpy array containing the times (indices) of discontinuities
	eeg_path = "/home/sebgho/eeg_project/data/subjects/npy/" + subject + ".npy"
	disc_path = "/home/sebgho/eeg_project/data/subjects/npy/" + subject + "_disc.npy"
	try:
		eeg_data = np.load(eeg_path)
	except:
		continue
	try:
		disc_data = np.load(disc_path)
	except:
		disc_data = np.array([])

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
	np.save("/home/sebgho/eeg_project/data/subjects/npy/" + subject + "_snippets.npy", snippet_array)
