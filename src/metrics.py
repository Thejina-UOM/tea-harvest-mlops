"""Model quality metrics shared by evaluation, the quality gate and monitoring."""

import numpy as np
from sklearn.metrics import (
    accuracy_score,
    average_precision_score,
    balanced_accuracy_score,
    f1_score,
    recall_score,
    roc_auc_score,
)


def score(y_true, y_prob, threshold: float = 0.5) -> dict:
    """Every metric I track for the tea readiness model, from predicted probabilities."""
    y_true = np.asarray(y_true).astype(int)
    y_prob = np.asarray(y_prob, dtype=float)
    y_pred = (y_prob >= threshold).astype(int)
    both_classes = len(np.unique(y_true)) == 2
    return {
        "accuracy": float(accuracy_score(y_true, y_pred)),
        "balanced_accuracy": float(balanced_accuracy_score(y_true, y_pred)),
        "f1": float(f1_score(y_true, y_pred, zero_division=0)),
        "recall_not_ready": float(recall_score(y_true, y_pred, pos_label=0, zero_division=0)),
        "roc_auc": float(roc_auc_score(y_true, y_prob)) if both_classes else float("nan"),
        "pr_auc": float(average_precision_score(y_true, y_prob)) if both_classes else float("nan"),
        "positive_rate": float(np.mean(y_true)),
        "mean_prediction": float(np.mean(y_prob)),
    }
