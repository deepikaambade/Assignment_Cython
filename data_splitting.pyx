# data_splitting.pyx

import numpy as np
cimport numpy as np
from sklearn.model_selection import train_test_split

cpdef tuple split_data(np.ndarray[np.float64_t, ndim=2] X, np.ndarray[np.float64_t] y, float test_size=0.2):
    """
    Function to split data into train and test sets.

    Parameters:
    - X (np.ndarray): Features.
    - y (np.ndarray): Target.
    - test_size (float): Size of the test set (default: 0.2).

    Returns:
    - tuple: Tuple containing train and test sets (X_train, X_test, y_train, y_test).
    """
    cdef np.ndarray[np.float64_t, ndim=2] X_train, X_test
    cdef np.ndarray[np.float64_t] y_train, y_test
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=test_size, random_state=42)
    return X_train, X_test, y_train, y_test
