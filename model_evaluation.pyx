# model_evaluation.pyx

import numpy as np
cimport numpy as np
from sklearn.metrics import mean_squared_error

cpdef float evaluate_model(LinearRegression model, np.ndarray[np.float64_t, ndim=2] X_test, np.ndarray[np.float64_t] y_test):
    """
    Function to evaluate the trained model.

    Parameters:
    - model (LinearRegression): Trained linear regression model.
    - X_test (np.ndarray): Features of the test set.
    - y_test (np.ndarray): Target of the test set.

    Returns:
    - float: Mean squared error of the model.
    """
    cdef np.ndarray[np.float64_t] y_pred = model.predict(X_test)
    return mean_squared_error(y_test, y_pred)
