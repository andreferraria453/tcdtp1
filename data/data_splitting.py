from sklearn.model_selection import train_test_split, KFold
def split_data_tt(X, y, test_size=0.3, random_state=10):
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=test_size, random_state=random_state, stratify=y
    )
    return X_train, X_test, y_train, y_test

def split_data_tvt(X, y, val_size=0.3, test_size=0.3, random_state=10):
    X_temp, X_test, y_temp, y_test = train_test_split(
        X, y, test_size=test_size, random_state=random_state, stratify=y
    )
    
    val_ratio = val_size / (1 - test_size)
    X_train, X_val, y_train, y_val = train_test_split(
        X_temp, y_temp, test_size=val_ratio, random_state=random_state, stratify=y_temp
    )
    
    return X_train, X_val, X_test, y_train, y_val, y_test


def split_data_kfold(X, n_splits, random, shuffle=True):
    kf = KFold(n_splits=n_splits, random_state=random, shuffle=shuffle)
    folds = []
    for train_idx, test_idx in kf.split(X):
        folds.append((train_idx, test_idx))
    return folds