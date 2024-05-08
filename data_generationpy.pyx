# data_generation.py

import numpy as np

def generate_data(num_samples):
    np.random.seed(0)
    X = np.random.rand(num_samples, 1) * 10
    y = 2 * X.squeeze() + np.random.randn(num_samples) * 2
    return X, y
