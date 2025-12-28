import numpy as np
import pandas as pd

def detect_outliers_iqr(data, scaling_factor=1.5):
    """Detecta outliers usando o método IQR"""
    Q1 = data.quantile(0.25)
    Q3 = data.quantile(0.75)
    IQR = Q3 - Q1
    
    lower_limit = Q1 - scaling_factor * IQR
    upper_limit = Q3 + scaling_factor * IQR
    
    mask_outliers = (data < lower_limit) | (data > upper_limit)
    density = np.sum(mask_outliers) / len(data) * 100
    
    return density, mask_outliers

def z_score(data):
    """Calcula z-scores"""
    mean = np.mean(data)
    std_dev = np.std(data)
    z_scores = (data - mean) / std_dev
    return z_scores, mean, std_dev

def identify_outliers_z_score(data, k=3):
    """Identifica outliers usando z-score"""
    z_scores, mean, std_dev = z_score(data)
    return np.abs(z_scores) > k, mean, std_dev

def inject_outliers(data, x, k, z):
    """Injeta outliers em um conjunto de dados para atingir uma densidade desejada de outliers."""
    data = np.array(data, dtype=float)
    print(f"inside")
    # Force 2D
    if data.ndim == 1:
        data = data.reshape(-1, 1)

    # define os limites originais e identifica outliers originais
    are_outliers, mean, std_dev = identify_outliers_z_score(data, k=3)
    
    n_total = len(are_outliers)
    n_outliers = np.sum(are_outliers)
    d = (n_outliers / n_total) * 100  # percentagem de outliers

    print(f"Density of outliers: {d}% | Desired outliers: {x}%")

    # Se já temos mais outliers do que o desejado, não injeta, mas retorna a máscara
    if d >= x:
        z_scores = (data - mean) / std_dev  
        mask_outliers = np.abs(z_scores)>k
        print("No new outliers injected.")
        return mask_outliers, data

    # Calcula quantos novos outliers são necessários
    ratio_missing_outliers = (x - d) / 100
    non_outlier_idx = np.where(~are_outliers)[0] # devolve o indice
    n_new_points = int(round(ratio_missing_outliers * n_total))

    # Sorteia índices para injetar os outliers
    new_idx = np.random.choice(non_outlier_idx, n_new_points, replace=False)
    s = np.random.choice([-1, 1], size=(n_new_points, 1))
    q = np.random.uniform(0, z, size=(n_new_points, 1))
    u = np.mean(data, axis=0)
    dev = np.std(data, axis=0)
    data[new_idx] = u + s * k * (dev + q)

    # Calcula a máscara final usando os limites originais
    z_scores = (data - mean) / std_dev  
    mask_outliers = np.abs(z_scores)>k


    n_outliers_final = np.sum(mask_outliers)
    d_final = (n_outliers_final / n_total) * 100
    print(f"Final outlier density: {d_final}%")

    return mask_outliers, data
