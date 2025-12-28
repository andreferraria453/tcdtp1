import numpy as np
import pandas as pd
from config.constants import COLUMNS_ACCELEROMETER, COLUMNS_GYROSCOPE, COLUMNS_MAGNETOMETER
def add_module_columns(dataset):
    """Adiciona colunas de módulo para cada sensor"""
    def calculate_module(columns, dataset, name):
        module = np.sqrt(np.sum(dataset[columns]**2, axis=1))
        dataset[name] = module
    calculate_module(COLUMNS_MAGNETOMETER, dataset, "magnetometer_module")
    calculate_module(COLUMNS_GYROSCOPE, dataset, "gyroscope_module")
    calculate_module(COLUMNS_ACCELEROMETER, dataset, "accelerometer_module")
    return dataset

def create_sliding_window(data, window_size):
    """Cria janelas deslizantes para séries temporais"""
    assert isinstance(data, np.ndarray), "Erro: 'data' deve ser um numpy.ndarray."
    X, y = [], []
    for i in range(len(data) - window_size):
        X.append(data[i:i + window_size])
        y.append(data[i + window_size])
    return np.array(X), np.array(y).reshape(-1, 1)