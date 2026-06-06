"""
Train TF-IDF + Logistic Regression baseline on Jigsaw train.csv and export model.pkl.

Usage (from repo root):
  python -m ml.training.train_baseline
  python -m ml.training.train_baseline --data data/raw/train.csv --metrics-out ml/experiments/baseline_tfidf_lr/metrics.json
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
from ml.labels import DEFAULT_THRESHOLD, LABELS
from ml.training.baseline_pipeline import build_baseline_pipeline


def _repo_root() -> Path:
    return Path(__file__).resolve().parents[2]


def _default_data_path() -> Path:
    return _repo_root() / "data" / "raw" / "train.csv"


def _tiny_demo_corpus() -> tuple[list[str], np.ndarray]:
    texts = [
        "Thank you, this helped a lot!",
        "You are stupid and I hate you",
        "I will find you and hurt you",
        "What the **** is wrong with you",
        "People like you should not exist",
        "Great article, very insightful",
        "Shut up idiot nobody cares",
        "Lovely weather today",
        "go to hell you id!ot",
        "1diot stop spamming",
    ]
    y = np.array(
        [
            [0, 0, 0, 0, 0, 0],
            [1, 0, 0, 0, 1, 0],
            [1, 0, 0, 1, 0, 0],
            [1, 0, 1, 0, 1, 0],
            [1, 1, 0, 0, 0, 1],
            [0, 0, 0, 0, 0, 0],
            [1, 0, 0, 0, 1, 0],
            [0, 0, 0, 0, 0, 0],
            [1, 0, 0, 0, 1, 0],
            [1, 0, 0, 0, 1, 0],
        ],
        dtype=np.int32,
    )
    return texts, y


def load_jigsaw_dataset(path: Path) -> tuple[list[str], np.ndarray]:
    """Load Kaggle Jigsaw train.csv → (texts, n×6 label matrix)."""
    df = pd.read_csv(path)
    missing_labels = [label for label in LABELS if label not in df.columns]
    if missing_labels:
        raise ValueError(f"Dataset is missing label columns: {missing_labels}")
    if "comment_text" not in df.columns:
        raise ValueError("Dataset must contain a 'comment_text' column.")

    texts = df["comment_text"].fillna("").astype(str).tolist()
    y = df[list(LABELS)].astype(np.int32).to_numpy()
    return texts, y


def _positive_class_scores(proba_item: np.ndarray) -> np.ndarray:
    """Extract P(y=1) from a single OvR binary predict_proba block."""
    arr = np.asarray(proba_item)
    if arr.ndim == 2 and arr.shape[1] >= 2:
        return arr[:, 1]
    if arr.ndim == 2 and arr.shape[1] == 1:
        return arr[:, 0]
    return arr.ravel()


def probabilities_to_predictions(proba: list[np.ndarray] | np.ndarray, threshold: float) -> np.ndarray:
    """Convert OneVsRest predict_proba output to binary predictions."""
    arr = np.asarray(proba)
    if arr.ndim == 2 and arr.shape[1] == len(LABELS):
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
    # Stratify by first label (toxic) for balanced hold-out
    stratify = y[:, 0]
    x_train, x_test, y_train, y_test = train_test_split(
        texts,
        y,
        test_size=test_size,
        random_state=random_state,
        stratify=stratify,
    )

    model = build_baseline_pipeline(random_state=random_state)
    model.fit(x_train, y_train)

    proba = model.predict_proba(x_test)
    y_pred = probabilities_to_predictions(proba, threshold)
    metrics = multilabel_report(y_test, y_pred, list(LABELS))
    metrics["dataset"] = {
        "n_samples": len(texts),
        "n_train": len(x_train),
        "n_test": len(x_test),
        "test_size": test_size,
        "threshold": threshold,
        "positive_rate_any_label": float((y.max(axis=1) > 0).mean()),
    }
    return model, metrics


def main() -> int:
    parser = argparse.ArgumentParser(
        description="Train TF-IDF + Logistic Regression baseline for toxic comments.",
    )
    parser.add_argument(
        "--data",
        type=Path,
        default=_default_data_path(),
        help="Path to Jigsaw train.csv (falls back to demo corpus if missing).",
    )
    parser.add_argument(
        "--out",
        type=Path,
        default=_repo_root() / "models" / "model.pkl",
        help="Output pickle path for the FastAPI backend.",
    )
    parser.add_argument(
        "--metrics-out",
        type=Path,
        default=_repo_root() / "ml" / "experiments" / "baseline_tfidf_lr" / "metrics.json",
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
    parser.add_argument(
        "--demo",
        action="store_true",
        help="Force the tiny demo corpus even if train.csv exists.",
    )
    args = parser.parse_args()

    if args.demo or not args.data.is_file():
        if not args.demo:
            print(f"Dataset not found at {args.data}; using demo corpus.", file=sys.stderr)
        texts, y = _tiny_demo_corpus()
        model = build_baseline_pipeline(random_state=args.random_state)
        model.fit(texts, y)
        metrics = {
            "mode": "demo",
            "dataset": {"n_samples": len(texts)},
            "note": "Demo corpus only — metrics are not meaningful.",
        }
    else:
        print(f"Loading dataset from {args.data} ...", flush=True)
        texts, y = load_jigsaw_dataset(args.data)
        print(f"Training on {len(texts):,} comments ...", flush=True)
        model, metrics = train_and_evaluate(
            texts,
            y,
            test_size=args.test_size,
            random_state=args.random_state,
            threshold=args.threshold,
        )
        metrics["mode"] = "jigsaw"
        metrics["data_path"] = str(args.data.resolve())
        print("\n=== Hold-out evaluation ===")
        print(f"Hamming loss: {metrics['hamming_loss']:.4f}")
        print(f"F1 macro:     {metrics['f1_macro']:.4f}")
        print(f"F1 micro:     {metrics['f1_micro']:.4f}")
        print("\nPer-label F1:")
        for row in metrics["per_label"]:
            print(f"  {row['label']:14s}  F1={row['f1']:.4f}  P={row['precision']:.4f}  R={row['recall']:.4f}")

    args.out.parent.mkdir(parents=True, exist_ok=True)
    with args.out.open("wb") as f:
        pickle.dump(model, f)
    print(f"\nSaved model to {args.out}")

    args.metrics_out.parent.mkdir(parents=True, exist_ok=True)
    payload = {
        "trained_at": datetime.now(UTC).isoformat(),
        "labels": list(LABELS),
        "model_path": str(args.out.resolve()),
        **metrics,
    }
    args.metrics_out.write_text(json.dumps(payload, indent=2, ensure_ascii=False), encoding="utf-8")
    print(f"Saved metrics to {args.metrics_out}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
