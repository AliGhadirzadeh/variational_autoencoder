import sys
import numpy as np

txt_path = sys.argv[1] + "_transpose"
subj = np.loadtxt(txt_path, skiprows=1)
# Convert time to seconds
subj[:, 0] = subj[:, 0] / 1000
npy_path = sys.argv[1] + ".npy"
np.save(npy_path, subj)