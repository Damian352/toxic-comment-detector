"""Registry of inference backends (TF-IDF+LR and BERT) for English and Polish."""

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
    """
    Central inference model registry.

    Holds two TF-IDF and two BERT instances (EN + PL), tracks loaded flags,
    and routes predict_proba by (ModelId, lang).
    """

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

    def __init__(self, tfidf_path: Path, bert_dir: Path, tfidf_path_pl: Path, bert_dir_pl: Path) -> None:
        self._tfidf = ToxicInferenceService(tfidf_path, lang="en")
        self._bert = BertInferenceService(bert_dir, lang="en")
        self._tfidf_pl = ToxicInferenceService(tfidf_path_pl, lang="pl")
        self._bert_pl = BertInferenceService(bert_dir_pl, lang="pl")

        self._loaded: dict[tuple[ModelId, str], bool] = {
            (ModelId.TFIDF_LR, "en"): False,
            (ModelId.BERT, "en"): False,
            (ModelId.BOTH, "en"): False,
            (ModelId.TFIDF_LR, "pl"): False,
            (ModelId.BERT, "pl"): False,
            (ModelId.BOTH, "pl"): False,
        }
        self._paths = {
            ModelId.TFIDF_LR: tfidf_path,
            ModelId.BERT: bert_dir,
            ModelId.BOTH: tfidf_path,
        }
        self._paths_pl = {
            ModelId.TFIDF_LR: tfidf_path_pl,
            ModelId.BERT: bert_dir_pl,
            ModelId.BOTH: tfidf_path_pl,
        }

    def load_all(self) -> None:
        """Try to load all artifacts; FileNotFoundError → loaded=False."""
        # English
        for model_id, service in (
            (ModelId.TFIDF_LR, self._tfidf),
            (ModelId.BERT, self._bert),
        ):
            try:
                service.load()
                self._loaded[(model_id, "en")] = True
            except FileNotFoundError:
                self._loaded[(model_id, "en")] = False
        
        self._loaded[(ModelId.BOTH, "en")] = self._loaded[(ModelId.TFIDF_LR, "en")] and self._loaded[(ModelId.BERT, "en")]

        # Polish — BOTH is available only when both backends are loaded
        for model_id, service in (
            (ModelId.TFIDF_LR, self._tfidf_pl),
            (ModelId.BERT, self._bert_pl),
        ):
            try:
                service.load()
                self._loaded[(model_id, "pl")] = True
            except FileNotFoundError:
                self._loaded[(model_id, "pl")] = False

        self._loaded[(ModelId.BOTH, "pl")] = self._loaded[(ModelId.TFIDF_LR, "pl")] and self._loaded[(ModelId.BERT, "pl")]

    def get_service(self, model_id: ModelId, lang: str = "en") -> Predictor:
        if not self._loaded.get((model_id, lang), False):
            path = self._paths_pl[model_id] if lang == "pl" else self._paths[model_id]
            raise FileNotFoundError(
                f"Model '{model_id.value}' ({lang}) is not loaded. Artifact: {path}"
            )
        if lang == "pl":
            if model_id is ModelId.TFIDF_LR:
                return self._tfidf_pl
            return self._bert_pl
        else:
            if model_id is ModelId.TFIDF_LR:
                return self._tfidf
            return self._bert

    def predict_proba(self, text: str, model_id: ModelId, lang: str = "en") -> dict[str, float]:
        return self.get_service(model_id, lang).predict_proba(text)

    def list_models(self, lang: str = "en") -> list[ModelInfo]:
        out: list[ModelInfo] = []
        for model_id, (name, description) in self.MODEL_CATALOG.items():
            path = self._paths_pl[model_id] if lang == "pl" else self._paths[model_id]
            out.append(
                ModelInfo(
                    id=model_id,
                    name=name,
                    description=description,
                    loaded=self._loaded.get((model_id, lang), False),
                    artifact_path=str(path),
                )
            )
        return out

    def is_loaded(self, model_id: ModelId, lang: str = "en") -> bool:
        return self._loaded.get((model_id, lang), False)

    @property
    def any_loaded(self) -> bool:
        return any(self._loaded.values())
