"""Registry of inference backends (TF-IDF+LR and BERT)."""

from __future__ import annotations

from dataclasses import dataclass
from enum import Enum
from pathlib import Path
from typing import Protocol

from app.services.bert_inference import BertInferenceService
from app.services.inference import ToxicInferenceService


class ModelId(str, Enum):
    TFIDF_LR = "tfidf_lr"
    BERT = "bert"
    BOTH = "both"


class Predictor(Protocol):
    @property
    def labels(self) -> tuple[str, ...]: ...

    def predict_proba(self, text: str) -> dict[str, float]: ...


@dataclass(frozen=True)
class ModelInfo:
    id: ModelId
    name: str
    description: str
    loaded: bool
    artifact_path: str


class InferenceRegistry:
    """Loads available models at startup and routes predictions by model id."""

    MODEL_CATALOG: dict[ModelId, tuple[str, str]] = {
        ModelId.TFIDF_LR: (
            "TF-IDF + Logistic Regression",
            "Sparse word/char n-grams with One-vs-Rest logistic regression (sklearn).",
        ),
        ModelId.BERT: (
            "BERT",
            "Fine-tuned transformer encoder with multi-label sigmoid head (Hugging Face).",
        ),
        ModelId.BOTH: (
            "Dual Comparison Mode",
            "Run predictions on both TF-IDF and BERT models simultaneously for direct comparison.",
        ),
    }

    def __init__(self, tfidf_path: Path, bert_dir: Path) -> None:
        self._tfidf = ToxicInferenceService(tfidf_path)
        self._bert = BertInferenceService(bert_dir)
        self._loaded: dict[ModelId, bool] = {
            ModelId.TFIDF_LR: False,
            ModelId.BERT: False,
            ModelId.BOTH: False,
        }
        self._paths = {
            ModelId.TFIDF_LR: tfidf_path,
            ModelId.BERT: bert_dir,
            ModelId.BOTH: tfidf_path, # just a placeholder
        }

    def load_all(self) -> None:
        for model_id, service in (
            (ModelId.TFIDF_LR, self._tfidf),
            (ModelId.BERT, self._bert),
        ):
            try:
                service.load()
                self._loaded[model_id] = True
            except FileNotFoundError:
                self._loaded[model_id] = False
        
        self._loaded[ModelId.BOTH] = self._loaded[ModelId.TFIDF_LR] and self._loaded[ModelId.BERT]

    def get_service(self, model_id: ModelId) -> Predictor:
        if not self._loaded.get(model_id, False):
            raise FileNotFoundError(
                f"Model '{model_id.value}' is not loaded. Artifact: {self._paths[model_id]}"
            )
        if model_id is ModelId.TFIDF_LR:
            return self._tfidf
        return self._bert

    def predict_proba(self, text: str, model_id: ModelId) -> dict[str, float]:
        return self.get_service(model_id).predict_proba(text)

    def list_models(self) -> list[ModelInfo]:
        out: list[ModelInfo] = []
        for model_id, (name, description) in self.MODEL_CATALOG.items():
            out.append(
                ModelInfo(
                    id=model_id,
                    name=name,
                    description=description,
                    loaded=self._loaded.get(model_id, False),
                    artifact_path=str(self._paths[model_id]),
                )
            )
        return out

    @property
    def any_loaded(self) -> bool:
        return any(self._loaded.values())
