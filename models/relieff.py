import numpy as np
def reliefF(X, y, n_neighbors=10):
    n_samples, n_features = X.shape
    scores = np.zeros(n_features)
    
    classes, counts = np.unique(y, return_counts=True)
    P_c = {cls: count / n_samples for cls, count in zip(classes, counts)}
    
    for idx in range(n_samples):
        sample = X[idx]
        label = y[idx]
        
        dists = np.linalg.norm(X - sample, axis=1)
        dists[idx] = np.inf  # ignorar a própria amostra
        
        hits_idx = np.argsort(dists[y == label])[:n_neighbors]
        hits = X[y == label][hits_idx]
        
        for cls in classes:
            if cls == label:
                continue
            miss_idx = np.argsort(dists[y == cls])[:n_neighbors]
            misses = X[y == cls][miss_idx]
            scores += np.mean(np.abs(sample - misses), axis=0) * P_c[cls]  # aumenta importância
        scores -= np.mean(np.abs(sample - hits), axis=0)  # diminui importância
    
    scores = (scores - scores.min()) / (scores.max() - scores.min())
    return scores