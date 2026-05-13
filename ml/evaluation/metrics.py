"""Standard multilabel classification metrics for experiments and notebooks."""

from __future__ import annotations

import numpy as np
from sklearn.metrics import (
    classification_report,
    f1_score,
    hamming_loss,
    precision_recall_fscore_support,
    precision_score,
    recall_score,
)


def multilabel_report(y_true: np.ndarray, y_pred: np.ndarray, label_names: list[str]) -> dict:
    """Return macro/micro metrics plus per-label precision/recall/F1."""
    y_true = np.asarray(y_true)
    y_pred = np.asarray(y_pred)
    out: dict = {
        "hamming_loss": float(hamming_loss(y_true, y_pred)),
        "f1_macro": float(f1_score(y_true, y_pred, average="macro", zero_division=0)),
        "f1_micro": float(f1_score(y_true, y_pred, average="micro", zero_division=0)),
        "precision_macro": float(precision_score(y_true, y_pred, average="macro", zero_division=0)),
        "precision_micro": float(precision_score(y_true, y_pred, average="micro", zero_division=0)),
        "recall_macro": float(recall_score(y_true, y_pred, average="macro", zero_division=0)),
        "recall_micro": float(recall_score(y_true, y_pred, average="micro", zero_division=0)),
    }
    p, r, f, support = precision_recall_fscore_support(
        y_true, y_pred, average=None, zero_division=0, labels=list(range(len(label_names)))
    )
    out["per_label"] = [
        {
            "label": label_names[i],
            "precision": float(p[i]),
            "recall": float(r[i]),
            "f1": float(f[i]),
            "support": int(support[i]),
        }
        for i in range(len(label_names))
    ]
    out["classification_report"] = classification_report(
        y_true, y_pred, target_names=label_names, zero_division=0
    )
    return out
