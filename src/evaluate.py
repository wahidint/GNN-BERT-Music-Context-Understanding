"""Evaluation metrics utilities."""

from sklearn.metrics import accuracy_score, f1_score, average_precision_score

def evaluate_predictions(y_true, y_pred, probs=None):
    result = {
        "accuracy": accuracy_score(y_true, y_pred),
        "macro_f1": f1_score(y_true, y_pred, average="macro")
    }
    if probs is not None:
        result["auc_pr"] = average_precision_score(y_true, probs)
    return result
