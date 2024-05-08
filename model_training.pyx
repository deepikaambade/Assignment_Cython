# model_training.pyx

import numpy as np
cimport numpy as np
from sklearn.linear_model import LinearRegression

cpdef LinearRegression train_model(np.ndarray[np.float64_t, ndim=2] X_train, np.ndarray[np.float64_t] y_train):
    """
    Function to train a linear regression model.

    Parameters:
    - X_train (np.ndarray): Features of the training set.
    - y_train (np.ndarray): Target of the training set.

    Returns:
    - LinearRegression: Trained linear regression model.
    """
    cdef LinearRegression model = LinearRegression()
    model.fit(X_train, y_train)
    return model
