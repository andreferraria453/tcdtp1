
import numpy as np
def pca(X, n_components=110):
    X_normalized = (X - np.mean(X, axis=0)) / np.std(X, axis=0)
    
    # Calcular a matriz de covariância
    cov_mat = np.cov(X_normalized, rowvar=False)
    
    # Calcular autovalores e autovetores
    eigenvalues, eigenvectors = np.linalg.eigh(cov_mat)
    # Ordenar por autovalores decrescentes
    sorted_idx = np.argsort(eigenvalues)[::-1]
    eigenvalues = eigenvalues[sorted_idx]
    eigenvectors = eigenvectors[:, sorted_idx]    
    # Selecionar os n componentes principais
    eigenvectors_subset = eigenvectors[:, :n_components]
    
    # Projetar os dados
    X_reduced = np.dot(X_normalized, eigenvectors_subset)
    
    return X_reduced, eigenvalues, eigenvectors