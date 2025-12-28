import numpy as np
class LinearRegression:
    def __init__(self, x, y):
        self.x = x
        self.y = y
        self.betas = np.zeros((x.shape[1] + 1, 1))
        self.losses = []

    def compute_mse(self, x, y):
        """Calcula o Mean Squared Error"""
        X_b = np.c_[np.ones((x.shape[0], 1)), x]
        y_pred = X_b @ self.betas
        errors = y_pred - y
        mse = np.sum(errors ** 2) / (2 * len(y))
        return mse

    def fit(self):
        """Ajusta o modelo usando equação normal"""
        X_b = np.c_[np.ones((self.x.shape[0], 1)), self.x]
        beta_full = np.linalg.pinv(X_b.T @ X_b) @ X_b.T @ self.y
        self.betas = beta_full
        return self.betas

    def predict(self, x):
        """Faz previsões"""
        X_b = np.c_[np.ones((x.shape[0], 1)), x]
        y_pred = X_b @ self.betas
        return y_pred