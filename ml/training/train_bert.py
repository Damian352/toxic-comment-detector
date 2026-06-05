"""
Fine-tune BERT for multi-label toxic comment classification and export to models/bert/.

Usage (from repo root):
  python -m ml.training.train_bert
  python -m ml.training.train_bert --demo --epochs 3
  python -m ml.training.train_bert --max-samples 5000 --epochs 1
"""

from __future__ import annotations

import argparse
import json
import sys
from datetime import UTC, datetime
from pathlib import Path

import numpy as np
import torch
from sklearn.model_selection import train_test_split
from transformers import AutoTokenizer, Trainer, TrainingArguments

from ml.evaluation.metrics import multilabel_report
from ml.labels import DEFAULT_THRESHOLD, LABELS
from ml.training.bert_multilabel import (
    DEFAULT_MAX_LENGTH,
    DEFAULT_PRETRAINED,
    ToxicCommentDataset,
    build_bert_model,
    logits_to_probabilities,
    probabilities_to_predictions,
    save_bert_artifact,
)
from ml.training.train_baseline import _default_data_path, _tiny_demo_corpus, load_jigsaw_dataset


def _repo_root() -> Path:
    return Path(__file__).resolve().parents[2]


def _compute_metrics_builder(threshold: float):
    def compute_metrics(eval_pred) -> dict[str, float]:
        logits, labels = eval_pred
        proba = logits_to_probabilities(logits)
        y_pred = probabilities_to_predictions(proba, threshold)
        y_true = np.asarray(labels).astype(np.int32)
        report = multilabel_report(y_true, y_pred, list(LABELS))
        return {
            "f1_macro": report["f1_macro"],
            "f1_micro": report["f1_micro"],
            "hamming_loss": report["hamming_loss"],
        }

    return compute_metrics


def train_and_evaluate(
    texts: list[str],
    y: np.ndarray,
    *,
    pretrained_name: str,
    out_dir: Path,
    test_size: float,
    random_state: int,
    threshold: float,
    max_length: int,
    epochs: float,
    batch_size: int,
    learning_rate: float,
    max_samples: int | None,
) -> dict:
    """Fine-tune BERT and return evaluation metrics on the hold-out split."""
    if max_samples is not None and max_samples < len(texts):
        rng = np.random.default_rng(random_state)
        idx = rng.choice(len(texts), size=max_samples, replace=False)
        texts = [texts[i] for i in idx]
        y = y[idx]

    stratify = y[:, 0]
    x_train, x_test, y_train, y_test = train_test_split(
        texts,
        y,
        test_size=test_size,
        random_state=random_state,
        stratify=stratify,
    )

    tokenizer = AutoTokenizer.from_pretrained(pretrained_name)
    model = build_bert_model(pretrained_name)

    train_ds = ToxicCommentDataset(x_train, y_train, tokenizer, max_length=max_length)
    eval_ds = ToxicCommentDataset(x_test, y_test, tokenizer, max_length=max_length)

    training_args = TrainingArguments(
        output_dir=str(out_dir / "checkpoints"),
        num_train_epochs=epochs,
        per_device_train_batch_size=batch_size,
        per_device_eval_batch_size=batch_size,
        learning_rate=learning_rate,
        weight_decay=0.01,
        eval_strategy="epoch",
        save_strategy="no",
        logging_steps=50,
        report_to=[],
        seed=random_state,
        fp16=torch.cuda.is_available(),
    )

    trainer = Trainer(
        model=model,
        args=training_args,
        train_dataset=train_ds,
        eval_dataset=eval_ds,
        compute_metrics=_compute_metrics_builder(threshold),
    )
    trainer.train()

    eval_out = trainer.predict(eval_ds)
    proba = logits_to_probabilities(eval_out.predictions)
    y_pred = probabilities_to_predictions(proba, threshold)
    metrics = multilabel_report(y_test, y_pred, list(LABELS))
    metrics["dataset"] = {
        "n_samples": len(texts),
        "n_train": len(x_train),
        "n_test": len(x_test),
        "test_size": test_size,
        "threshold": threshold,
        "max_length": max_length,
        "pretrained_name": pretrained_name,
        "positive_rate_any_label": float((y.max(axis=1) > 0).mean()),
    }
    if max_samples is not None:
        metrics["dataset"]["max_samples"] = max_samples

    save_bert_artifact(
        trainer.model,
        tokenizer,
        out_dir,
        metadata={
            "pretrained_name": pretrained_name,
            "max_length": max_length,
            "trained_at": datetime.now(UTC).isoformat(),
        },
    )
    return metrics


def main() -> int:
    parser = argparse.ArgumentParser(description="Fine-tune BERT for multi-label toxic comments.")
    parser.add_argument("--data", type=Path, default=_default_data_path())
    parser.add_argument("--out", type=Path, default=_repo_root() / "models" / "bert")
    parser.add_argument(
        "--metrics-out",
        type=Path,
        default=_repo_root() / "ml" / "experiments" / "bert_multilabel" / "metrics.json",
    )
    parser.add_argument("--pretrained", type=str, default=DEFAULT_PRETRAINED)
    parser.add_argument("--test-size", type=float, default=0.2)
    parser.add_argument("--threshold", type=float, default=DEFAULT_THRESHOLD)
    parser.add_argument("--random-state", type=int, default=42)
    parser.add_argument("--max-length", type=int, default=DEFAULT_MAX_LENGTH)
    parser.add_argument("--epochs", type=float, default=1.0)
    parser.add_argument("--batch-size", type=int, default=16)
    parser.add_argument("--learning-rate", type=float, default=2e-5)
    parser.add_argument(
        "--max-samples",
        type=int,
        default=None,
        help="Subsample the dataset (useful for quick experiments on CPU).",
    )
    parser.add_argument("--demo", action="store_true")
    args = parser.parse_args()

    if args.demo or not args.data.is_file():
        if not args.demo:
            print(f"Dataset not found at {args.data}; using demo corpus.", file=sys.stderr)
        texts, y = _tiny_demo_corpus()
        tokenizer = AutoTokenizer.from_pretrained(args.pretrained)
        model = build_bert_model(args.pretrained)
        train_ds = ToxicCommentDataset(texts, y, tokenizer, max_length=args.max_length)
        training_args = TrainingArguments(
            output_dir=str(args.out / "checkpoints"),
            num_train_epochs=max(args.epochs, 3.0),
            per_device_train_batch_size=min(args.batch_size, len(texts)),
            learning_rate=args.learning_rate,
            save_strategy="no",
            report_to=[],
            seed=args.random_state,
        )
        Trainer(model=model, args=training_args, train_dataset=train_ds).train()
        save_bert_artifact(
            model,
            tokenizer,
            args.out,
            metadata={"mode": "demo", "pretrained_name": args.pretrained},
        )
        metrics = {
            "mode": "demo",
            "dataset": {"n_samples": len(texts)},
            "note": "Demo corpus only — metrics are not meaningful.",
        }
    else:
        print(f"Loading dataset from {args.data} ...", flush=True)
        texts, y = load_jigsaw_dataset(args.data)
        print(f"Fine-tuning BERT on {len(texts):,} comments ...", flush=True)
        metrics = train_and_evaluate(
            texts,
            y,
            pretrained_name=args.pretrained,
            out_dir=args.out,
            test_size=args.test_size,
            random_state=args.random_state,
            threshold=args.threshold,
            max_length=args.max_length,
            epochs=args.epochs,
            batch_size=args.batch_size,
            learning_rate=args.learning_rate,
            max_samples=args.max_samples,
        )
        metrics["mode"] = "jigsaw"
        metrics["data_path"] = str(args.data.resolve())
        print("\n=== Hold-out evaluation ===")
        print(f"Hamming loss: {metrics['hamming_loss']:.4f}")
        print(f"F1 macro:     {metrics['f1_macro']:.4f}")
        print(f"F1 micro:     {metrics['f1_micro']:.4f}")

    args.metrics_out.parent.mkdir(parents=True, exist_ok=True)
    payload = {
        "trained_at": datetime.now(UTC).isoformat(),
        "labels": list(LABELS),
        "model_dir": str(args.out.resolve()),
        "approach": "bert_multilabel",
        **metrics,
    }
    args.metrics_out.write_text(json.dumps(payload, indent=2, ensure_ascii=False), encoding="utf-8")
    print(f"\nSaved BERT artifact to {args.out}")
    print(f"Saved metrics to {args.metrics_out}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
