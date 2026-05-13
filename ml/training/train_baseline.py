"""
Train a small TF-IDF + One-vs-Rest baseline and export a pickle for the FastAPI backend.

Usage (from repo root):
  python -m ml.training.train_baseline

Or from ml/ after installing requirements:
  python training/train_baseline.py
"""

from __future__ import annotations

import argparse
import pickle
import sys
from pathlib import Path

import numpy as np
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.multiclass import OneVsRestClassifier
from sklearn.pipeline import Pipeline

# Keep labels in the same order as backend/app/services/inference.py:LABELS
LABELS: tuple[str, ...] = (
    "toxic",
    "severe_toxic",
    "obscene",
    "threat",
    "insult",
    "identity_hate",
)


def _repo_root() -> Path:
    return Path(__file__).resolve().parents[2]


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
        ],
        dtype=np.int32,
    )
    return texts, y


def build_model() -> Pipeline:
    clf = OneVsRestClassifier(
        LogisticRegression(max_iter=2000, class_weight="balanced", random_state=42),
    )
    return Pipeline(
        [
            (
                "tfidf",
                TfidfVectorizer(
                    lowercase=True,
                    ngram_range=(1, 2),
                    min_df=1,
                    max_features=20_000,
                ),
            ),
            ("clf", clf),
        ]
    )


def main() -> int:
    parser = argparse.ArgumentParser(description="Train baseline toxic comment model (demo data).")
    parser.add_argument(
        "--out",
        type=Path,
        default=_repo_root() / "models" / "model.pkl",
        help="Output pickle path (backend loads this by default).",
    )
    args = parser.parse_args()

    texts, y = _tiny_demo_corpus()
    model = build_model()
    model.fit(texts, y)

    args.out.parent.mkdir(parents=True, exist_ok=True)
    with args.out.open("wb") as f:
        pickle.dump(model, f)

    print(f"Saved model to {args.out}")
    print("Labels:", ", ".join(LABELS))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
