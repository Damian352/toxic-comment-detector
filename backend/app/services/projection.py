"""
Load precomputed validation-set PCA projections and project live user comments.

Artifacts live under `models/projections/{lang}/{model}/`:
  - reducer.joblib — fitted TruncatedSVD+PCA or PCA + display normalization meta
  - corpus.json    — validation points with error_type for scatter plots
"""

from __future__ import annotations

import json
import pickle
import sys
from pathlib import Path
from typing import Any

import numpy as np
from scipy import sparse

from app.services.inference import ToxicInferenceService
from app.services.registry import InferenceRegistry, ModelId

_REPO_ROOT = Path(__file__).resolve().parents[3]
if str(_REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(_REPO_ROOT))

from ml.labels import active_labels_from_probs, get_per_label_thresholds  # noqa: E402
from ml.visualization.embedding_projection import (  # noqa: E402
    cosine_similarity,
    extract_bert_cls_embeddings,
    project_embedding,
)


class ProjectionService:
    """Serves corpus scatter data and projects user comments into the same PCA space."""

    def __init__(self, projections_dir: Path) -> None:
        self._dir = projections_dir
        self._corpus_cache: dict[tuple[str, str], dict[str, Any]] = {}
        self._reducer_cache: dict[tuple[str, str], dict[str, Any]] = {}

    def is_available(self, lang: str, model_kind: str) -> bool:
        return (self._dir / lang / model_kind / "corpus.json").is_file()

    def _load_corpus(self, lang: str, model_kind: str) -> dict[str, Any]:
        key = (lang, model_kind)
        if key not in self._corpus_cache:
            path = self._dir / lang / model_kind / "corpus.json"
            if not path.is_file():
                raise FileNotFoundError(f"Projection corpus not found: {path}")
            self._corpus_cache[key] = json.loads(path.read_text(encoding="utf-8"))
        return self._corpus_cache[key]

    def _load_reducer(self, lang: str, model_kind: str) -> dict[str, Any]:
        key = (lang, model_kind)
        if key not in self._reducer_cache:
            path = self._dir / lang / model_kind / "reducer.joblib"
            if not path.is_file():
                raise FileNotFoundError(f"Projection reducer not found: {path}")
            with path.open("rb") as f:
                self._reducer_cache[key] = pickle.load(f)
        return self._reducer_cache[key]

    def get_corpus(
        self,
        lang: str,
        model_kind: str,
        *,
        error_filter: str | None = None,
    ) -> dict[str, Any]:
        """Return validation-set points and projection metadata for the UI."""
        corpus = self._load_corpus(lang, model_kind)
        points = list(corpus["points"])
        if error_filter and error_filter != "all":
            if error_filter == "errors":
                points = [p for p in points if p.get("error_type") != "correct"]
            else:
                points = [p for p in points if p.get("error_type") == error_filter]

        return {
            "lang": corpus["lang"],
            "model": corpus["model"],
            "method": corpus["method"],
            "explained_variance_ratio": corpus.get("explained_variance_ratio", []),
            "axes": corpus.get("axes", {"x": "PC1", "y": "PC2", "z": "PC3"}),
            "n_total_test": corpus.get("n_total_test", len(points)),
            "n_displayed": len(points),
            "error_counts": corpus.get("error_counts", {}),
            "points": points,
        }

    def _bert_dir(self, registry: InferenceRegistry, lang: str) -> Path:
        return registry._paths_pl[ModelId.BERT] if lang == "pl" else registry._paths[ModelId.BERT]

    def _embed_text(
        self,
        text: str,
        lang: str,
        model_kind: str,
        registry: InferenceRegistry,
    ) -> np.ndarray | sparse.csr_matrix:
        model_id = ModelId.TFIDF_LR if model_kind == "tfidf_lr" else ModelId.BERT
        if model_kind == "tfidf_lr":
            service = registry.get_service(model_id, lang)
            if not isinstance(service, ToxicInferenceService):
                raise RuntimeError("Expected ToxicInferenceService")
            return service.embed(text)
        bert_dir = self._bert_dir(registry, lang)
        return extract_bert_cls_embeddings(bert_dir, [text])

    def _embed_texts_batch(
        self,
        texts: list[str],
        lang: str,
        model_kind: str,
        registry: InferenceRegistry,
    ) -> list[np.ndarray | sparse.csr_matrix]:
        if not texts:
            return []
        model_id = ModelId.TFIDF_LR if model_kind == "tfidf_lr" else ModelId.BERT
        if model_kind == "tfidf_lr":
            service = registry.get_service(model_id, lang)
            if not isinstance(service, ToxicInferenceService):
                raise RuntimeError("Expected ToxicInferenceService")
            service.ensure_loaded()
            assert service._model is not None
            from ml.visualization.embedding_projection import extract_tfidf_embeddings

            mat = extract_tfidf_embeddings(service._model, texts)
            return [mat[i] for i in range(mat.shape[0])]
        bert_dir = self._bert_dir(registry, lang)
        mat = extract_bert_cls_embeddings(bert_dir, texts)
        return [mat[i] for i in range(mat.shape[0])]

    def build_user_projection(
        self,
        text: str,
        probs: dict[str, float],
        lang: str,
        model_kind: str,
        registry: InferenceRegistry,
        *,
        top_k: int = 5,
    ) -> tuple[dict[str, Any], list[dict[str, Any]]]:
        """Project user text and return (active_point, nearest_validation_neighbors)."""
        reducer = self._load_reducer(lang, model_kind)
        corpus = self._load_corpus(lang, model_kind)
        labels = tuple(reducer.get("labels", []))
        thresholds = get_per_label_thresholds(lang, model_kind, labels)
        pred_labels = active_labels_from_probs(probs, thresholds, lang)

        embedding = self._embed_text(text, lang, model_kind, registry)
        coords = project_embedding(reducer, embedding)
        x, y = float(coords[0]), float(coords[1])
        z = float(coords[2]) if len(coords) > 2 else 0.0

        active = {
            "id": "active_user",
            "text": text,
            "labels": pred_labels,
            "x": x,
            "y": y,
            "z": z,
            "ground_truth_labels": [],
            "predicted_labels": pred_labels,
            "error_type": None,
            "is_active": True,
            "is_validation": False,
            "similarity": 1.0,
        }

        corpus_points = list(corpus["points"])
        try:
            corpus_embs = self._embed_texts_batch(
                [p["text"] for p in corpus_points],
                lang,
                model_kind,
                registry,
            )
            user_vec = embedding.toarray().ravel() if sparse.issparse(embedding) else np.asarray(embedding).ravel()
            for idx, c_emb in enumerate(corpus_embs):
                c_vec = c_emb.toarray().ravel() if sparse.issparse(c_emb) else np.asarray(c_emb).ravel()
                corpus_points[idx] = {
                    **corpus_points[idx],
                    "similarity": cosine_similarity(user_vec, c_vec),
                }
        except Exception:
            pass

        neighbors = sorted(
            [p for p in corpus_points if not p.get("is_active")],
            key=lambda p: p.get("similarity", 0.0),
            reverse=True,
        )[:top_k]
        return active, neighbors
