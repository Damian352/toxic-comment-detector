"""BERT-based multi-label toxic comment inference (Hugging Face)."""

from __future__ import annotations

import json
import sys
from pathlib import Path
from typing import Any

import numpy as np
import torch
from transformers import AutoModelForSequenceClassification, AutoTokenizer

_REPO_ROOT = Path(__file__).resolve().parents[3]

if str(_REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(_REPO_ROOT))

from ml.labels import LABELS, PL_LABELS  # noqa: E402
from ml.training.bert_multilabel import DEFAULT_MAX_LENGTH, logits_to_probabilities  # noqa: E402


class BertInferenceService:
    """Loads a fine-tuned BERT directory and returns sigmoid probabilities per label."""

    def __init__(self, model_dir: Path, lang: str = "en") -> None:
        self._model_dir = model_dir
        self._model: Any | None = None
        self._tokenizer: Any | None = None
        self._max_length = DEFAULT_MAX_LENGTH
        self._lang = lang
        self._labels: tuple[str, ...] = PL_LABELS if lang == "pl" else LABELS
        self._device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    @property
    def labels(self) -> tuple[str, ...]:
        return self._labels

    def load(self) -> None:
        if not self._model_dir.is_dir():
            raise FileNotFoundError(
                f"BERT model directory not found: {self._model_dir}. "
                "Train with: python -m ml.training.train_bert"
            )
        config_path = self._model_dir / "labels.json"
        if config_path.is_file():
            meta = json.loads(config_path.read_text(encoding="utf-8"))
            self._max_length = int(meta.get("max_length", DEFAULT_MAX_LENGTH))
            if "labels" in meta:
                self._labels = tuple(meta["labels"])

        self._tokenizer = AutoTokenizer.from_pretrained(self._model_dir)
        self._model = AutoModelForSequenceClassification.from_pretrained(self._model_dir)
        self._model.to(self._device)
        self._model.eval()

    def ensure_loaded(self) -> None:
        if self._model is None or self._tokenizer is None:
            self.load()

    def predict_proba(self, text: str) -> dict[str, float]:
        self.ensure_loaded()
        assert self._model is not None and self._tokenizer is not None

        encoding = self._tokenizer(
            text,
            truncation=True,
            padding="max_length",
            max_length=self._max_length,
            return_tensors="pt",
        )
        encoding = {key: value.to(self._device) for key, value in encoding.items()}

        with torch.no_grad():
            outputs = self._model(**encoding)
            logits = outputs.logits.cpu().numpy()

        scores = logits_to_probabilities(logits)[0]
        if scores.shape[0] != len(self.labels):
            raise ValueError(
                f"Model output size {scores.shape[0]} does not match labels ({len(self.labels)})."
            )
        return {label: float(score) for label, score in zip(self.labels, scores, strict=True)}
