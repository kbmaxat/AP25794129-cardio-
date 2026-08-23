from __future__ import annotations

import numpy as np
from sklearn.metrics import accuracy_score, confusion_matrix, precision_recall_fscore_support, roc_auc_score


def binary_metrics(labels, probabilities, threshold: float = 0.5) -> dict[str, float | int]:
    y_true = np.asarray(labels, dtype=int)
    y_score = np.asarray(probabilities, dtype=float)
    y_pred = (y_score >= threshold).astype(int)
    tn, fp, fn, tp = confusion_matrix(y_true, y_pred, labels=[0, 1]).ravel()
    precision, recall, f1, _ = precision_recall_fscore_support(
        y_true, y_pred, average="binary", zero_division=0
    )
    specificity = tn / (tn + fp) if tn + fp else 0.0
    auc = roc_auc_score(y_true, y_score) if len(np.unique(y_true)) == 2 else float("nan")
    return {
        "accuracy": float(accuracy_score(y_true, y_pred)),
        "precision": float(precision),
        "recall": float(recall),
        "specificity": float(specificity),
        "f1": float(f1),
        "roc_auc": float(auc),
        "tn": int(tn), "fp": int(fp), "fn": int(fn), "tp": int(tp),
    }
