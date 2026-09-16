import numpy as np
from sklearn.metrics import (accuracy_score, classification_report,
                             confusion_matrix, f1_score, precision_score,
                             recall_score, roc_auc_score)


def get_confusionmatrix_fnd(preds, labels):
    print(confusion_matrix(labels, preds, labels=[0, 1]))


def metrics(y_true, y_pred, y_probs=None):
    metrics = {}
    y_true = np.array(y_true)
    y_pred = np.array(y_pred)

    if y_probs is not None:
        y_probs = np.array(y_probs)
        try:
            metrics['auc'] = roc_auc_score(y_true, y_probs, average='macro')
        except ValueError:
            metrics['auc'] = 0.5
    else:
        metrics['auc'] = 0.0

    metrics['f1'] = f1_score(y_true, y_pred, average='macro')
    metrics['recall'] = recall_score(y_true, y_pred, average='macro')
    metrics['precision'] = precision_score(y_true, y_pred, average='macro')
    metrics['acc'] = accuracy_score(y_true, y_pred)
    metrics['preds'] = y_pred.tolist()
    metrics['labels'] = y_true.tolist()

    return metrics
