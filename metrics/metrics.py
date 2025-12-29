from sklearn.metrics import confusion_matrix, recall_score, precision_score, f1_score,accuracy_score
import numpy as np
import pandas as pd
def compute_confusion_matrix(y_true, y_pred, labels=None):
    unique_in_data = np.unique(y_true)

    if labels is not None and len(labels) > 0 and isinstance(labels[0], str) and not isinstance(y_true[0], str):
        cm = confusion_matrix(y_true, y_pred, labels=range(len(labels)))
        display_classes = labels
    else:
        cm = confusion_matrix(y_true, y_pred, labels=labels)
        display_classes = labels if labels is not None else unique_in_data
    
    return pd.DataFrame(cm, 
                        index=[f"Real_{l}" for l in display_classes], 
                        columns=[f"Pred_{l}" for l in display_classes])



def compute_recall(y_true, y_pred, average='macro'):
    return recall_score(y_true, y_pred, average=average,zero_division=0)


def compute_precision(y_true, y_pred, average='weighted'):
    return precision_score(y_true, y_pred, average=average,zero_division=0)


def compute_f1_score(y_true, y_pred, average='weighted'):
    return f1_score(y_true, y_pred, average=average,zero_division=0)

def compute_accuracy(y_true, y_pred):
    return accuracy_score(y_true, y_pred)
