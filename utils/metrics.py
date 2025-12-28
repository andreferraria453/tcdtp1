import numpy as np
from models.linear_regression import LinearRegression

def gcv_linear_regression(X, y):
    """Calcula Generalized Cross Validation para regressão linear"""
    n, p = X.shape
    
    model = LinearRegression(X, y)
    model.fit()
    
    y_pred = model.predict(X)
    residuals = y.flatten() - y_pred.flatten()
    rss = np.sum(residuals**2)
    
    XtX_inv = np.linalg.inv(X.T @ X)
    hat_matrix_trace = np.sum((X @ XtX_inv) * X)
    
    gcv = rss / (1 - hat_matrix_trace / n)**2
    return gcv, model