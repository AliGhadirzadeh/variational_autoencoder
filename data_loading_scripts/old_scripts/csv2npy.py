import sys
import numpy as np
import pandas as pd

csv_path = sys.argv[1]
data = pd.read_csv(csv_path, sep='\t')
times = data.values[:, 2]
times = times.astype(float)
# Convert time to seconds
times /= 256
npy_path = csv_path[:-4] + ".npy"
np.save(npy_path, times)