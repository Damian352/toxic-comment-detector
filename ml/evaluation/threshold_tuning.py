"""Per-label probability threshold tuning for multi-label classifiers."""

from __future__ import annotations

from typing import Any

import numpy as np
from sklearn.metrics import f1_score

from ml.evaluation.metrics import multilabel_report


def default_threshold_grid() -> np.ndarray:
    """Coarse grid for threshold search (inclusive 0.05 .. 0.95)."""
    return np.round(np.arange(0.05, 0.96, 0.05), 2)


def tune_per_label_thresholds(
    y_true: np.ndarray,
    y_proba: np.ndarray,
    label_names: list[str],
    *,
    grid: np.ndarray | None = None,
) -> dict[str, float]:
    """
    Pick a threshold per label that maximizes binary F1 on the given split.

    Each label is tuned independently — standard for imbalanced multi-label setups.
    """
    y_true = np.asarray(y_true, dtype=np.int32)
    y_proba = np.asarray(y_proba, dtype=np.float64)
    if grid is None:
        grid = default_threshold_grid()

    thresholds: dict[str, float] = {}
    per_label_f1: dict[str, float] = {}

    for i, label in enumerate(label_names):
        best_t = 0.5
        best_f1 = -1.0
        for t in grid:
            y_pred = (y_proba[:, i] >= t).astype(np.int32)
            f1 = float(f1_score(y_true[:, i], y_pred, zero_division=0))
            if f1 > best_f1:
                best_f1 = f1
                best_t = float(t)
        thresholds[label] = best_t
        per_label_f1[label] = best_f1

    return thresholds


def apply_per_label_thresholds(
    y_proba: np.ndarray,
    thresholds: dict[str, float],
    label_names: list[str],
) -> np.ndarray:
    """Convert probability matrix to binary predictions using per-label thresholds."""
    y_proba = np.asarray(y_proba, dtype=np.float64)
    tvec = np.array([thresholds[label] for label in label_names], dtype=np.float64)
    return (y_proba >= tvec).astype(np.int32)


def evaluate_with_thresholds(
    y_true: np.ndarray,
    y_proba: np.ndarray,
    label_names: list[str],
    thresholds: dict[str, float] | float,
) -> dict[str, Any]:
    """Build a multilabel_report using either uniform or per-label thresholds."""
    if isinstance(thresholds, (int, float)):
        y_pred = (y_proba >= float(thresholds)).astype(np.int32)
        threshold_payload: dict[str, Any] = {"uniform": float(thresholds)}
    else:
        y_pred = apply_per_label_thresholds(y_proba, thresholds, label_names)
        threshold_payload = {"per_label": thresholds}

    report = multilabel_report(y_true, y_pred, label_names)
    report["thresholds"] = threshold_payload
    return report


def compare_threshold_reports(
    baseline: dict[str, Any],
    tuned: dict[str, Any],
) -> dict[str, Any]:
    """Summarize macro/micro metric deltas between two evaluation reports."""
    return {
        "f1_macro_delta": tuned["f1_macro"] - baseline["f1_macro"],
        "f1_micro_delta": tuned["f1_micro"] - baseline["f1_micro"],
        "precision_macro_delta": tuned["precision_macro"] - baseline["precision_macro"],
        "recall_macro_delta": tuned["recall_macro"] - baseline["recall_macro"],
        "hamming_loss_delta": tuned["hamming_loss"] - baseline["hamming_loss"],
        "baseline_f1_macro": baseline["f1_macro"],
        "tuned_f1_macro": tuned["f1_macro"],
        "baseline_f1_micro": baseline["f1_micro"],
        "tuned_f1_micro": tuned["f1_micro"],
    }
