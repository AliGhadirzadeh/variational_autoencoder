import numpy as np

# Load subject snippet data in list
subject_list = []
for i in range(1, 52):
	subject_list.append(str(i))

data_list = []

for subject in subject_list:
	subject_path = "./data/snippets/" +  subject + "_snippets.npy"
	try:
		subject_data = np.load(subject_path)
	except:
		continue
	# Add subjects with complete data ONLY
	if subject_data.shape[2] == 20:
		data_list.append(subject_data)

# Initiate numpy array
number_of_snippets = sum(list(map(lambda sub : sub.shape[0], data_list)))
snippet_length = data_list[0].shape[1]
snippet_channels = 20
snippet_array = np.zeros((number_of_snippets, snippet_length, snippet_channels))

# Add data to numpy array and save
start_index = 0
end_index = 0
for subject_data in data_list:
	start_index = end_index
	end_index += subject_data.shape[0]
	snippet_array[start_index:end_index] = subject_data

np.save("./data/snippets/snippets.npy", snippet_array)