"""
Train TF-IDF + Logistic Regression baseline on Polish BAN-PL.csv and export model_pl.pkl.

Usage (from repo root):
  python -m ml.training.train_baseline_pl
"""

from __future__ import annotations

import argparse
import json
import pickle
import sys
from datetime import UTC, datetime
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split

from ml.evaluation.metrics import multilabel_report
from ml.labels import DEFAULT_THRESHOLD, PL_LABELS
from ml.preprocessing.text import preprocess_batch_pl
from ml.training.baseline_pipeline import build_baseline_pipeline


def _repo_root() -> Path:
    return Path(__file__).resolve().parents[2]


def _default_data_path() -> Path:
    return _repo_root() / "BAN-PL_2" / "BAN-PL.csv"


def load_polish_dataset(path: Path) -> tuple[list[str], np.ndarray]:
    """Load Polish BAN-PL dataset into texts and multi-label matrix."""
    df = pd.read_csv(path)
    df = df.drop(columns=[col for col in df.columns if "Unnamed" in col], errors="ignore")
    
    # Preprocess
    df["CleanText"] = df["Text"].fillna("").apply(lambda t: " ".join(preprocess_batch_pl([str(t)])))
    
    # Pivot
    df["value"] = 1
    df_multi = df.pivot_table(
        index="CleanText",
        columns="Reason",
        values="value",
        fill_value=0
    ).reset_index()

    # Re-order columns if needed, columns must be [1, 2, 3, 4]
    cols = [1, 2, 3, 4]
    for c in cols:
        if c not in df_multi.columns:
            df_multi[c] = 0

    texts = df_multi["CleanText"].tolist()
    y = df_multi[cols].astype(np.int32).to_numpy()
    return texts, y


def _positive_class_scores(proba_item: np.ndarray) -> np.ndarray:
    arr = np.asarray(proba_item)
    if arr.ndim == 2 and arr.shape[1] >= 2:
        return arr[:, 1]
    if arr.ndim == 2 and arr.shape[1] == 1:
        return arr[:, 0]
    return arr.ravel()


def probabilities_to_predictions(proba: list[np.ndarray] | np.ndarray, threshold: float) -> np.ndarray:
    arr = np.asarray(proba)
    if arr.ndim == 2 and arr.shape[1] == len(PL_LABELS):
        scores = arr
    else:
        scores = np.column_stack([_positive_class_scores(p) for p in proba])
    return (scores >= threshold).astype(np.int32)


def train_and_evaluate(
    texts: list[str],
    y: np.ndarray,
    *,
    test_size: float,
    random_state: int,
    threshold: float,
) -> tuple[object, dict]:
    """Fit baseline pipeline and return the model plus evaluation metrics."""
    stratify = y[:, 0]  # Stratify by safety label
    x_train, x_test, y_train, y_test = train_test_split(
        texts,
        y,
        test_size=test_size,
        random_state=random_state,
        stratify=stratify,
    )

    model = build_baseline_pipeline(
        random_state=random_state,
        preprocess_func=preprocess_batch_pl
    )
    model.fit(x_train, y_train)

    proba = model.predict_proba(x_test)
    y_pred = probabilities_to_predictions(proba, threshold)
    metrics = multilabel_report(y_test, y_pred, list(PL_LABELS))
    metrics["dataset"] = {
        "n_samples": len(texts),
        "n_train": len(x_train),
        "n_test": len(x_test),
        "test_size": test_size,
        "threshold": threshold,
        "positive_rate_any_label": float((y[:, 1:].max(axis=1) > 0).mean()), # toxic rates (categories 2,3,4)
    }
    return model, metrics


def main() -> int:
    parser = argparse.ArgumentParser(
        description="Train TF-IDF + Logistic Regression baseline for Polish comments.",
    )
    parser.add_argument(
        "--data",
        type=Path,
        default=_default_data_path(),
        help="Path to Polish BAN-PL.csv",
    )
    parser.add_argument(
        "--out",
        type=Path,
        default=_repo_root() / "models" / "model_pl.pkl",
        help="Output pickle path for Polish TF-IDF baseline.",
    )
    parser.add_argument(
        "--metrics-out",
        type=Path,
        default=_repo_root() / "ml" / "experiments" / "baseline_tfidf_lr_pl" / "metrics.json",
        help="Where to save evaluation metrics as JSON.",
    )
    parser.add_argument("--test-size", type=float, default=0.2, help="Hold-out test fraction.")
    parser.add_argument(
        "--threshold",
        type=float,
        default=DEFAULT_THRESHOLD,
        help="Probability threshold for binary predictions in evaluation.",
    )
    parser.add_argument("--random-state", type=int, default=42)
    args = parser.parse_args()

    if not args.data.is_file():
        print(f"Error: Dataset not found at {args.data}.", file=sys.stderr)
        return 1

    print(f"Loading Polish dataset from {args.data}...", flush=True)
    texts, y = load_polish_dataset(args.data)
    print(f"Loaded {len(texts)} samples.", flush=True)

    print("Training and evaluating baseline pipeline...", flush=True)
    model, metrics = train_and_evaluate(
        texts,
        y,
        test_size=args.test_size,
        random_state=args.random_state,
        threshold=args.threshold,
    )

    print(f"Hamming Loss: {metrics['hamming_loss']:.5f}")
    print(f"Macro F1:     {metrics['f1_macro']:.5f}")
    print(f"Micro F1:     {metrics['f1_micro']:.5f}")

    args.out.parent.mkdir(parents=True, exist_ok=True)
    with args.out.open("wb") as f:
        pickle.dump(model, f)
    print(f"Exported serialized pipeline to {args.out}")

    args.metrics_out.parent.mkdir(parents=True, exist_ok=True)
    metrics["trained_at"] = datetime.now(UTC).isoformat()
    metrics["model_path"] = str(args.out.resolve())
    with args.metrics_out.open("w", encoding="utf-8") as f:
        json.dump(metrics, f, indent=2, ensure_ascii=False)
    print(f"Saved evaluation metrics to {args.metrics_out}")

    return 0


if __name__ == "__main__":
    sys.exit(main())
