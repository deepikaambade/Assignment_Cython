# data_generation.pyx

import numpy as np
cimport numpy as np

cpdef tuple generate_data(int num_samples):
    """
    Function to generate synthetic data.

    Parameters:
    - num_samples (int): Number of samples to generate.

    Returns:
    - tuple: Tuple containing features (X) and target (y).
    """
    np.random.seed(0)
    cdef np.ndarray[np.float64_t, ndim=2] X = np.random.rand(num_samples, 1) * 10
    cdef np.ndarray[np.float64_t] y = 2 * X.squeeze() + np.random.randn(num_samples) * 2
    return X, y
