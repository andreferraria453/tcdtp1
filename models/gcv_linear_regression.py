#gcv

import numpy as np
from sklearn.linear_model import LinearRegression


def gcv_linear_regression(X, y):
    n, p = X.shape

    model = LinearRegression()
    model.fit(X,y)
    
    y_pred = model.predict(X)
    residuals = y.flatten() - y_pred.flatten()
    rss = np.sum(residuals**2) # Residual Sum of Squares (RSS)
    
    
    XtX_inv = np.linalg.inv(X.T @ X)
    hat_matrix_trace = np.sum((X @ XtX_inv) * X)
    
    gcv = rss / (1 - hat_matrix_trace / n)**2

    return gcv, model