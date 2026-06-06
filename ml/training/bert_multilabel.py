"""BERT multi-label classifier utilities for toxic comment detection."""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

import numpy as np
import torch
from torch.utils.data import Dataset
from transformers import AutoModelForSequenceClassification, AutoTokenizer, PreTrainedModel, PreTrainedTokenizerBase

from ml.labels import DEFAULT_THRESHOLD, LABELS

DEFAULT_PRETRAINED = "bert-base-uncased"
DEFAULT_MAX_LENGTH = 256


class ToxicCommentDataset(Dataset):
    """PyTorch dataset: tokenized comments with float multi-label targets."""

    def __init__(
        self,
        texts: list[str],
        labels: np.ndarray,
        tokenizer: PreTrainedTokenizerBase,
        *,
        max_length: int = DEFAULT_MAX_LENGTH,
    ) -> None:
        self.texts = texts
        self.labels = labels.astype(np.float32)
        self.tokenizer = tokenizer
        self.max_length = max_length

    def __len__(self) -> int:
        return len(self.texts)

    def __getitem__(self, idx: int) -> dict[str, torch.Tensor]:
        encoding = self.tokenizer(
            self.texts[idx],
            truncation=True,
            padding="max_length",
            max_length=self.max_length,
            return_tensors="pt",
        )
        item = {key: value.squeeze(0) for key, value in encoding.items()}
        item["labels"] = torch.tensor(self.labels[idx], dtype=torch.float32)
        return item


def build_bert_model(pretrained_name: str = DEFAULT_PRETRAINED) -> PreTrainedModel:
    """Load a BERT encoder with a multi-label classification head (sigmoid at inference)."""
    id2label = {i: label for i, label in enumerate(LABELS)}
    label2id = {label: i for i, label in enumerate(LABELS)}
    return AutoModelForSequenceClassification.from_pretrained(
        pretrained_name,
        num_labels=len(LABELS),
        problem_type="multi_label_classification",
        id2label=id2label,
        label2id=label2id,
    )


def save_bert_artifact(
    model: PreTrainedModel,
    tokenizer: PreTrainedTokenizerBase,
    out_dir: Path,
    *,
    metadata: dict[str, Any] | None = None,
) -> None:
    """Persist Hugging Face weights, tokenizer, and label order for the backend."""
    out_dir.mkdir(parents=True, exist_ok=True)
    model.save_pretrained(out_dir)
    tokenizer.save_pretrained(out_dir)
    payload = {
        "labels": list(LABELS),
        "default_threshold": DEFAULT_THRESHOLD,
        **(metadata or {}),
    }
    (out_dir / "labels.json").write_text(
        json.dumps(payload, indent=2, ensure_ascii=False),
        encoding="utf-8",
    )


def logits_to_probabilities(logits: np.ndarray) -> np.ndarray:
    """Apply sigmoid to raw logits (shape n_samples × n_labels or n_labels)."""
    arr = np.asarray(logits, dtype=np.float64)
    return 1.0 / (1.0 + np.exp(-arr))


def probabilities_to_predictions(proba: np.ndarray, threshold: float = DEFAULT_THRESHOLD) -> np.ndarray:
    """Threshold sigmoid probabilities into binary multi-label predictions."""
    return (np.asarray(proba) >= threshold).astype(np.int32)


def probabilities_to_predictions_per_label(
    proba: np.ndarray,
    thresholds: dict[str, float],
    label_names: list[str] | tuple[str, ...],
) -> np.ndarray:
    """Apply a separate decision threshold for each label."""
    arr = np.asarray(proba, dtype=np.float64)
    tvec = np.array([thresholds[label] for label in label_names], dtype=np.float64)
    if arr.ndim == 1:
        return (arr >= tvec).astype(np.int32)
    return (arr >= tvec).astype(np.int32)
