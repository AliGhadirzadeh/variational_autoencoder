import numpy as np
import queue as queue

# Initialize parameters and subjects
snippet_length = 500
window = 100
subject_list = []
for i in range(52):
	subject_list.append(str(i))

for subject in subject_list:
	# Load data, create queue
	# eeg_data is numpy array containing channel voltages over time
	# disc_data is a numpy array containing the times (indices) of discontinuities
	eeg_path = "/home/sebgho/eeg_project/data/" + subject + ".npy"
	disc_path = "/home/sebgho/eeg_project/data/" + subject + "_disc.npy"
	eeg_data = np.load(eeg_path)
	disc_data = np.load(disc_path)

	disc_queue = queue.Queue()
	for i in range(disc_data.shape[0]):
		disc_queue.put(disc_data[i])

	# Initialize times and snippet list
	t = 0
	t_max = eeg_data.shape[0]
	t_disc = disc_queue.get()
	snippet_list = []

	# Create snippets
	while t < t_max:
		# If no discontinuity, add snippet
		if t + snippet_length < t_disc:
			snippet_list.append(eeg_data[t:(t+snippet_length), :])
			t += window
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
	np.save("/home/sebgho/eeg_project/data/" + subject + "_snippets.npy", snippet_array)
