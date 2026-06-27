"""
Load precomputed validation-set PCA projections and project live user comments.

Artifacts live under `models/projections/{lang}/{model}/`:
  - reducer.joblib — fitted TruncatedSVD+PCA or PCA + display normalization meta
  - corpus.json    — validation points with error_type for scatter plots
  - corpus_embeddings.npz — precomputed embeddings for fast similarity lookup
"""

from __future__ import annotations

import json
import math
import pickle
import sys
from pathlib import Path
from typing import Any

import numpy as np
from scipy import sparse

from app.services.bert_inference import BertInferenceService
from app.services.inference import ToxicInferenceService
from app.services.registry import InferenceRegistry, ModelId

_REPO_ROOT = Path(__file__).resolve().parents[3]
if str(_REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(_REPO_ROOT))

from ml.labels import active_labels_from_probs, get_per_label_thresholds  # noqa: E402
from ml.visualization.embedding_projection import (  # noqa: E402
    cosine_similarity,
    load_corpus_embeddings,
    project_embedding,
)


class ProjectionService:
    """Serves corpus scatter data and projects user comments into the same PCA space."""

    def __init__(self, projections_dir: Path) -> None:
        self._dir = projections_dir
        self._corpus_cache: dict[tuple[str, str], dict[str, Any]] = {}
        self._reducer_cache: dict[tuple[str, str], dict[str, Any]] = {}
        self._corpus_embeddings_cache: dict[tuple[str, str], list[np.ndarray | sparse.csr_matrix]] = {}

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

    def _load_corpus_embeddings(self, lang: str, model_kind: str) -> list[np.ndarray | sparse.csr_matrix] | None:
        key = (lang, model_kind)
        if key not in self._corpus_embeddings_cache:
            path = self._dir / lang / model_kind / "corpus_embeddings.npz"
            if not path.is_file():
                return None
            _ids, vectors = load_corpus_embeddings(path)
            self._corpus_embeddings_cache[key] = vectors
        return self._corpus_embeddings_cache[key]

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
        service = registry.get_service(model_id, lang)
        if not isinstance(service, BertInferenceService):
            raise RuntimeError("Expected BertInferenceService")
        return service.extract_cls_embedding(text)

    @staticmethod
    def _coords_similarity(
        user_coords: tuple[float, float, float],
        point: dict[str, Any],
        *,
        sigma: float = 45.0,
    ) -> float:
        """Fallback similarity from PCA display coordinates (1.0 when positions coincide)."""
        ux, uy, uz = user_coords
        px, py = float(point["x"]), float(point["y"])
        pz = float(point.get("z", 0.0))
        dist_sq = (ux - px) ** 2 + (uy - py) ** 2 + (uz - pz) ** 2
        return float(math.exp(-dist_sq / (2.0 * sigma**2)))

    def _similarity_neighbors(
        self,
        user_embedding: np.ndarray | sparse.csr_matrix,
        corpus_points: list[dict[str, Any]],
        lang: str,
        model_kind: str,
        registry: InferenceRegistry,
        *,
        user_coords: tuple[float, float, float],
        user_text: str,
    ) -> list[dict[str, Any]]:
        """Attach cosine similarity to corpus points (embeddings, else PCA coordinate distance)."""
        user_text_norm = " ".join(user_text.split()).casefold()
        corpus_embs = self._load_corpus_embeddings(lang, model_kind)
        use_embeddings = corpus_embs is not None and len(corpus_embs) == len(corpus_points)

        user_vec: np.ndarray | None = None
        if use_embeddings:
            user_vec = (
                user_embedding.toarray().ravel()
                if sparse.issparse(user_embedding)
                else np.asarray(user_embedding).ravel()
            )

        def _preview_prefix(raw: str) -> str:
            cleaned = " ".join(raw.split()).casefold()
            if cleaned.endswith("..."):
                cleaned = cleaned[:-3].rstrip()
            return cleaned

        user_preview = _preview_prefix(user_text)

        enriched: list[dict[str, Any]] = []
        for idx, point in enumerate(corpus_points):
            point_preview = _preview_prefix(str(point.get("text", "")))
            if user_preview and point_preview and (
                user_preview == point_preview
                or user_preview.startswith(point_preview)
                or point_preview.startswith(user_preview)
            ):
                enriched.append({**point, "similarity": 1.0})
                continue

            if use_embeddings and user_vec is not None:
                c_emb = corpus_embs[idx]
                c_vec = c_emb.toarray().ravel() if sparse.issparse(c_emb) else np.asarray(c_emb).ravel()
                sim = cosine_similarity(user_vec, c_vec)
            else:
                sim = self._coords_similarity(user_coords, point)
            enriched.append({**point, "similarity": sim})
        return enriched

    def build_user_projection(
        self,
        text: str,
        probs: dict[str, float],
        lang: str,
        model_kind: str,
        registry: InferenceRegistry,
    ) -> tuple[dict[str, Any], list[dict[str, Any]]]:
        """Project user text and return (active_point, corpus_points_with_similarity)."""
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
        user_coords = (x, y, z)
        corpus_points = self._similarity_neighbors(
            embedding,
            corpus_points,
            lang,
            model_kind,
            registry,
            user_coords=user_coords,
            user_text=text,
        )
        return active, corpus_points
