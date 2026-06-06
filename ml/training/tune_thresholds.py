"""
Tune per-label probability thresholds and re-evaluate all four inference models.

Uses the same outer test split as training (test_size=0.2, random_state=42).
Thresholds are fit on an inner validation split (20% of train) and final metrics
are reported on the held-out test set.

Usage (from repo root):
  python -m ml.training.tune_thresholds
  python -m ml.training.tune_thresholds --skip-bert   # TF-IDF only (faster)
"""

from __future__ import annotations

import argparse
import json
import pickle
import sys
from datetime import UTC, datetime
from pathlib import Path

import numpy as np
import torch
from sklearn.model_selection import train_test_split
from transformers import AutoModelForSequenceClassification, AutoTokenizer

from ml.evaluation.metrics import multilabel_report
from ml.evaluation.threshold_tuning import (
    apply_per_label_thresholds,
    compare_threshold_reports,
    evaluate_with_thresholds,
    tune_per_label_thresholds,
)
from ml.labels import DEFAULT_THRESHOLD, LABELS, PL_LABELS
from ml.training.bert_multilabel import DEFAULT_MAX_LENGTH, logits_to_probabilities
from ml.training.train_baseline import (
    _positive_class_scores,
    load_jigsaw_dataset,
)
from ml.training.train_baseline_pl import load_polish_dataset


def _repo_root() -> Path:
    return Path(__file__).resolve().parents[2]


def _sklearn_proba_matrix(model: object, texts: list[str], n_labels: int) -> np.ndarray:
    """Run predict_proba and return shape (n_samples, n_labels)."""
    raw = model.predict_proba(texts)
    arr = np.asarray(raw)
    if arr.ndim == 2 and arr.shape[1] == n_labels:
        return arr.astype(np.float64)
    return np.column_stack([_positive_class_scores(p) for p in raw]).astype(np.float64)


def _bert_proba_matrix(
    model_dir: Path,
    texts: list[str],
    *,
    batch_size: int = 32,
) -> np.ndarray:
    """Batch inference for a saved Hugging Face multi-label directory."""
    labels_path = model_dir / "labels.json"
    max_length = DEFAULT_MAX_LENGTH
    if labels_path.is_file():
        meta = json.loads(labels_path.read_text(encoding="utf-8"))
        max_length = int(meta.get("max_length", DEFAULT_MAX_LENGTH))

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    tokenizer = AutoTokenizer.from_pretrained(model_dir)
    model = AutoModelForSequenceClassification.from_pretrained(model_dir)
    model.to(device)
    model.eval()

    outputs: list[np.ndarray] = []
    for start in range(0, len(texts), batch_size):
        batch = texts[start : start + batch_size]
        encoding = tokenizer(
            batch,
            truncation=True,
            padding=True,
            max_length=max_length,
            return_tensors="pt",
        )
        encoding = {k: v.to(device) for k, v in encoding.items()}
        with torch.no_grad():
            logits = model(**encoding).logits.cpu().numpy()
        outputs.append(logits_to_probabilities(logits))

    return np.vstack(outputs)


def _split_for_tuning(
    texts: list[str],
    y: np.ndarray,
    *,
    test_size: float,
    val_size: float,
    random_state: int,
) -> tuple[list[str], np.ndarray, list[str], np.ndarray, list[str], np.ndarray]:
    """Outer test split + inner validation split from train (matches training seed)."""
    stratify_outer = y[:, 0]
    x_train, x_test, y_train, y_test = train_test_split(
        texts,
        y,
        test_size=test_size,
        random_state=random_state,
        stratify=stratify_outer,
    )
    stratify_inner = y_train[:, 0]
    x_fit, x_val, y_fit, y_val = train_test_split(
        x_train,
        y_train,
        test_size=val_size,
        random_state=random_state,
        stratify=stratify_inner,
    )
    return x_fit, y_fit, x_val, y_val, x_test, y_test


def _tune_sklearn_model(
    *,
    name: str,
    model_path: Path,
    texts: list[str],
    y: np.ndarray,
    label_names: list[str],
    test_size: float,
    val_size: float,
    random_state: int,
    thresholds_out: Path,
    metrics_out: Path,
) -> dict:
    """Tune thresholds for a pickle sklearn model; saves thresholds + metrics JSON."""
    print(f"\n=== {name} ===", flush=True)
    if not model_path.is_file():
        raise FileNotFoundError(f"Model not found: {model_path}")

    with model_path.open("rb") as f:
        model = pickle.load(f)

    _x_fit, _y_fit, x_val, y_val, x_test, y_test = _split_for_tuning(
        texts,
        y,
        test_size=test_size,
        val_size=val_size,
        random_state=random_state,
    )

    print(f"  Tuning on val={len(x_val):,}, evaluating on test={len(x_test):,}", flush=True)
    proba_val = _sklearn_proba_matrix(model, x_val, len(label_names))
    thresholds = tune_per_label_thresholds(y_val, proba_val, label_names)

    proba_test = _sklearn_proba_matrix(model, x_test, len(label_names))
    baseline = evaluate_with_thresholds(y_test, proba_test, label_names, DEFAULT_THRESHOLD)
    tuned = evaluate_with_thresholds(y_test, proba_test, label_names, thresholds)
    comparison = compare_threshold_reports(baseline, tuned)

    print("  Per-label thresholds (F1-optimal on val):")
    for label in label_names:
        print(f"    {label:14s}  t={thresholds[label]:.2f}")

    print(f"  Test F1 macro: {baseline['f1_macro']:.4f} -> {tuned['f1_macro']:.4f} "
          f"(delta {comparison['f1_macro_delta']:+.4f})")
    print(f"  Test F1 micro: {baseline['f1_micro']:.4f} -> {tuned['f1_micro']:.4f} "
          f"(delta {comparison['f1_micro_delta']:+.4f})")

    payload = {
        "model": name,
        "model_path": str(model_path.resolve()),
        "labels": label_names,
        "tuned_at": datetime.now(UTC).isoformat(),
        "tuning": {
            "method": "per_label_max_f1",
            "grid": "0.05..0.95 step 0.05",
            "val_size": val_size,
            "test_size": test_size,
            "random_state": random_state,
        },
        "per_label_thresholds": thresholds,
        "default_threshold_baseline": DEFAULT_THRESHOLD,
        "evaluation_test": {
            "baseline_uniform_0.5": baseline,
            "tuned_per_label": tuned,
            "comparison": comparison,
        },
    }

    thresholds_out.parent.mkdir(parents=True, exist_ok=True)
    thresholds_out.write_text(json.dumps(payload, indent=2, ensure_ascii=False), encoding="utf-8")
    print(f"  Saved thresholds to {thresholds_out}")

    metrics_out.parent.mkdir(parents=True, exist_ok=True)
    metrics_payload = {
        "trained_at": payload["tuned_at"],
        "labels": label_names,
        "model_path": str(model_path.resolve()),
        "threshold": thresholds,
        "threshold_mode": "per_label",
        **{k: tuned[k] for k in (
            "hamming_loss", "f1_macro", "f1_micro",
            "precision_macro", "precision_micro",
            "recall_macro", "recall_micro",
            "per_label", "classification_report",
        )},
        "dataset": {
            "n_test": len(x_test),
            "test_size": test_size,
            "baseline_f1_macro": baseline["f1_macro"],
            "baseline_f1_micro": baseline["f1_micro"],
        },
        "baseline_at_0.5": {
            "f1_macro": baseline["f1_macro"],
            "f1_micro": baseline["f1_micro"],
            "per_label": baseline["per_label"],
        },
    }
    metrics_out.write_text(json.dumps(metrics_payload, indent=2, ensure_ascii=False), encoding="utf-8")
    print(f"  Updated metrics at {metrics_out}")

    return payload


def _tune_bert_model(
    *,
    name: str,
    model_dir: Path,
    texts: list[str],
    y: np.ndarray,
    label_names: list[str],
    test_size: float,
    val_size: float,
    random_state: int,
    thresholds_out: Path,
    metrics_out: Path,
    batch_size: int,
) -> dict:
    print(f"\n=== {name} ===", flush=True)
    if not model_dir.is_dir():
        raise FileNotFoundError(f"Model directory not found: {model_dir}")

    _x_fit, _y_fit, x_val, y_val, x_test, y_test = _split_for_tuning(
        texts,
        y,
        test_size=test_size,
        val_size=val_size,
        random_state=random_state,
    )

    print(f"  Tuning on val={len(x_val):,}, evaluating on test={len(x_test):,}", flush=True)
    proba_val = _bert_proba_matrix(model_dir, x_val, batch_size=batch_size)
    thresholds = tune_per_label_thresholds(y_val, proba_val, label_names)

    proba_test = _bert_proba_matrix(model_dir, x_test, batch_size=batch_size)
    baseline = evaluate_with_thresholds(y_test, proba_test, label_names, DEFAULT_THRESHOLD)
    tuned = evaluate_with_thresholds(y_test, proba_test, label_names, thresholds)
    comparison = compare_threshold_reports(baseline, tuned)

    print("  Per-label thresholds (F1-optimal on val):")
    for label in label_names:
        print(f"    {label:14s}  t={thresholds[label]:.2f}")

    print(f"  Test F1 macro: {baseline['f1_macro']:.4f} -> {tuned['f1_macro']:.4f} "
          f"(delta {comparison['f1_macro_delta']:+.4f})")
    print(f"  Test F1 micro: {baseline['f1_micro']:.4f} -> {tuned['f1_micro']:.4f} "
          f"(delta {comparison['f1_micro_delta']:+.4f})")

    # Merge thresholds into labels.json for backend discovery
    labels_json_path = model_dir / "labels.json"
    labels_meta: dict = {}
    if labels_json_path.is_file():
        labels_meta = json.loads(labels_json_path.read_text(encoding="utf-8"))
    labels_meta["per_label_thresholds"] = thresholds
    labels_meta["default_threshold"] = DEFAULT_THRESHOLD
    labels_meta["threshold_tuned_at"] = datetime.now(UTC).isoformat()
    labels_json_path.write_text(json.dumps(labels_meta, indent=2, ensure_ascii=False), encoding="utf-8")

    payload = {
        "model": name,
        "model_dir": str(model_dir.resolve()),
        "labels": label_names,
        "tuned_at": datetime.now(UTC).isoformat(),
        "tuning": {
            "method": "per_label_max_f1",
            "grid": "0.05..0.95 step 0.05",
            "val_size": val_size,
            "test_size": test_size,
            "random_state": random_state,
        },
        "per_label_thresholds": thresholds,
        "default_threshold_baseline": DEFAULT_THRESHOLD,
        "evaluation_test": {
            "baseline_uniform_0.5": baseline,
            "tuned_per_label": tuned,
            "comparison": comparison,
        },
    }

    thresholds_out.parent.mkdir(parents=True, exist_ok=True)
    thresholds_out.write_text(json.dumps(payload, indent=2, ensure_ascii=False), encoding="utf-8")
    print(f"  Saved thresholds to {thresholds_out}")

    metrics_out.parent.mkdir(parents=True, exist_ok=True)
    metrics_payload = {
        "trained_at": payload["tuned_at"],
        "labels": label_names,
        "model_dir": str(model_dir.resolve()),
        "threshold": thresholds,
        "threshold_mode": "per_label",
        **{k: tuned[k] for k in (
            "hamming_loss", "f1_macro", "f1_micro",
            "precision_macro", "precision_micro",
            "recall_macro", "recall_micro",
            "per_label", "classification_report",
        )},
        "dataset": {
            "n_test": len(x_test),
            "test_size": test_size,
            "baseline_f1_macro": baseline["f1_macro"],
            "baseline_f1_micro": baseline["f1_micro"],
        },
        "baseline_at_0.5": {
            "f1_macro": baseline["f1_macro"],
            "f1_micro": baseline["f1_micro"],
            "per_label": baseline["per_label"],
        },
    }
    metrics_out.write_text(json.dumps(metrics_payload, indent=2, ensure_ascii=False), encoding="utf-8")
    print(f"  Updated metrics at {metrics_out}")

    return payload


def main() -> int:
    root = _repo_root()
    parser = argparse.ArgumentParser(description="Tune per-label thresholds for all models.")
    parser.add_argument("--test-size", type=float, default=0.2)
    parser.add_argument("--val-size", type=float, default=0.2, help="Fraction of train used for threshold tuning.")
    parser.add_argument("--random-state", type=int, default=42)
    parser.add_argument("--batch-size", type=int, default=32)
    parser.add_argument("--skip-bert", action="store_true", help="Skip BERT/HerBERT (TF-IDF only).")
    parser.add_argument(
        "--en-data",
        type=Path,
        default=root / "data" / "raw" / "train.csv",
    )
    parser.add_argument(
        "--pl-data",
        type=Path,
        default=root / "BAN-PL_2" / "BAN-PL.csv",
    )
    args = parser.parse_args()

    summary: dict[str, dict] = {}

    # English TF-IDF
    if args.en_data.is_file():
        print(f"Loading English Jigsaw data from {args.en_data} ...", flush=True)
        en_texts, en_y = load_jigsaw_dataset(args.en_data)
        summary["en_tfidf_lr"] = _tune_sklearn_model(
            name="en_tfidf_lr",
            model_path=root / "models" / "model.pkl",
            texts=en_texts,
            y=en_y,
            label_names=list(LABELS),
            test_size=args.test_size,
            val_size=args.val_size,
            random_state=args.random_state,
            thresholds_out=root / "ml" / "experiments" / "baseline_tfidf_lr" / "thresholds.json",
            metrics_out=root / "ml" / "experiments" / "baseline_tfidf_lr" / "metrics.json",
        )
        if not args.skip_bert:
            summary["en_bert"] = _tune_bert_model(
                name="en_bert",
                model_dir=root / "models" / "bert",
                texts=en_texts,
                y=en_y,
                label_names=list(LABELS),
                test_size=args.test_size,
                val_size=args.val_size,
                random_state=args.random_state,
                thresholds_out=root / "ml" / "experiments" / "bert_multilabel" / "thresholds.json",
                metrics_out=root / "ml" / "experiments" / "bert_multilabel" / "metrics.json",
                batch_size=args.batch_size,
            )
    else:
        print(f"Skipping English models — dataset not found: {args.en_data}", file=sys.stderr)

    # Polish TF-IDF + HerBERT
    if args.pl_data.is_file():
        print(f"\nLoading Polish BAN-PL data from {args.pl_data} ...", flush=True)
        pl_texts, pl_y = load_polish_dataset(args.pl_data)
        summary["pl_tfidf_lr"] = _tune_sklearn_model(
            name="pl_tfidf_lr",
            model_path=root / "models" / "model_pl.pkl",
            texts=pl_texts,
            y=pl_y,
            label_names=list(PL_LABELS),
            test_size=args.test_size,
            val_size=args.val_size,
            random_state=args.random_state,
            thresholds_out=root / "ml" / "experiments" / "baseline_tfidf_lr_pl" / "thresholds.json",
            metrics_out=root / "ml" / "experiments" / "baseline_tfidf_lr_pl" / "metrics.json",
        )
        if not args.skip_bert:
            summary["pl_bert"] = _tune_bert_model(
                name="pl_bert",
                model_dir=root / "models" / "bert_pl",
                texts=pl_texts,
                y=pl_y,
                label_names=list(PL_LABELS),
                test_size=args.test_size,
                val_size=args.val_size,
                random_state=args.random_state,
                thresholds_out=root / "ml" / "experiments" / "bert_multilabel_pl" / "thresholds.json",
                metrics_out=root / "ml" / "experiments" / "bert_multilabel_pl" / "metrics.json",
                batch_size=args.batch_size,
            )
    else:
        print(f"Skipping Polish models — dataset not found: {args.pl_data}", file=sys.stderr)

    # Consolidated threshold registry for backend / API
    registry = {
        "tuned_at": datetime.now(UTC).isoformat(),
        "method": "per_label_max_f1",
        "en": {
            "tfidf_lr": summary.get("en_tfidf_lr", {}).get("per_label_thresholds"),
            "bert": summary.get("en_bert", {}).get("per_label_thresholds"),
        },
        "pl": {
            "tfidf_lr": summary.get("pl_tfidf_lr", {}).get("per_label_thresholds"),
            "bert": summary.get("pl_bert", {}).get("per_label_thresholds"),
        },
    }
    registry_path = root / "models" / "thresholds.json"
    registry_path.write_text(json.dumps(registry, indent=2, ensure_ascii=False), encoding="utf-8")
    print(f"\nSaved consolidated registry to {registry_path}")

    # Also save per-pickle sidecars for sklearn models
    for lang_key, pkl_name in (("en", "model.pkl"), ("pl", "model_pl.pkl")):
        key = f"{lang_key}_tfidf_lr" if lang_key == "en" else "pl_tfidf_lr"
        th = summary.get(key, {}).get("per_label_thresholds")
        if th:
            sidecar = root / "models" / pkl_name.replace(".pkl", "_thresholds.json")
            sidecar.write_text(json.dumps(th, indent=2), encoding="utf-8")

    print("\n=== Summary (test set, tuned vs baseline 0.5) ===")
    for key, payload in summary.items():
        cmp_ = payload["evaluation_test"]["comparison"]
        print(
            f"  {key:12s}  F1 macro {cmp_['baseline_f1_macro']:.4f} -> {cmp_['tuned_f1_macro']:.4f}  "
            f"(delta {cmp_['f1_macro_delta']:+.4f})"
        )

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
