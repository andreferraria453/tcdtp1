import numpy as np
from models.linear_regression import LinearRegression
import pandas as pd
from collections import defaultdict
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




def mean_std_confusion_by_model(cm_dict):
    model_matrices = defaultdict(list)
    for model_key, model_data in cm_dict.items():
        model_name = model_key.split("_{")[0]
        model_matrices[model_name].extend(model_data["all_matrices"])

    mean_by_model = {}
    std_by_model = {}

    # Calcula média e std por modelo
    for model_name, matrices in model_matrices.items():
        stacked = np.stack([df.values for df in matrices])

        mean_df = pd.DataFrame(
            stacked.mean(axis=0),
            index=matrices[0].index,
            columns=matrices[0].columns
        )

        std_df = pd.DataFrame(
            stacked.std(axis=0),
            index=matrices[0].index,
            columns=matrices[0].columns
        )

        mean_by_model[model_name] = mean_df
        std_by_model[model_name] = std_df

    return mean_by_model, std_by_model
