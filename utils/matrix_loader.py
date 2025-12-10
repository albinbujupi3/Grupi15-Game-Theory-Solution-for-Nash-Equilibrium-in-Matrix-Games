import numpy as np
import os

def load_matrix(path):
    if not os.path.exists(path):
        raise FileNotFoundError(path)
    if path.endswith('.npy'):
        return np.load(path)
    # try csv
    return np.loadtxt(path, delimiter=',')
