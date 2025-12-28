from sklearn.metrics import confusion_matrix, recall_score, precision_score, f1_score,accuracy_score
import numpy as np
import pandas as pd
def compute_confusion_matrix(y_true, y_pred, labels=None):
    cm = confusion_matrix(y_true, y_pred, labels=labels)
    classes = labels if labels is not None else np.unique(y_true)
    return pd.DataFrame(cm, 
                        index=[f"Real_{l}" for l in classes], 
                        columns=[f"Pred_{l}" for l in classes])




def compute_recall(y_true, y_pred, average='macro'):
    return recall_score(y_true, y_pred, average=average,zero_division=0)


def compute_precision(y_true, y_pred, average='weighted'):
    return precision_score(y_true, y_pred, average=average,zero_division=0)


def compute_f1_score(y_true, y_pred, average='weighted'):
    return f1_score(y_true, y_pred, average=average,zero_division=0)

def compute_accuracy(y_true, y_pred):
    return accuracy_score(y_true, y_pred)
