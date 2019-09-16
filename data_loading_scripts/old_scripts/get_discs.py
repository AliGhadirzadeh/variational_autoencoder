import numpy as np

# Script to generate mock discontiuity timelines

subject_list = []
for i in range(1, 52):
	subject_list.append(str(i))

for subject in subject_list:
	eeg_path = "/home/sebgho/eeg_project/data/subjects/npy/" + subject + ".npy"
	try:
		eeg_data = np.load(eeg_path)
	except:
		continue
	else:
		size = np.random.randint(0, 10, 1)
		disc_times = np.random.randint(0, eeg_data.shape[0], size)
		disc_times = np.sort(disc_times)

		disc_path = "/home/sebgho/eeg_project/data/subjects/npy/" + subject + "_disc.npy"
		np.save(disc_path, disc_times)
