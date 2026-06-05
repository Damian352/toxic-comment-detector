import pickle
import sys
from pathlib import Path
from typing import Any

import numpy as np

_REPO_ROOT = Path(__file__).resolve().parents[3]

if str(_REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(_REPO_ROOT))

from ml.labels import LABELS  # noqa: E402


def _scores_from_predict_proba(proba: Any, n_labels: int) -> np.ndarray:
    """Normalize sklearn predict_proba outputs to shape (n_labels,)."""
    if isinstance(proba, list):
        scores: list[float] = []
        for p in proba[:n_labels]:
            arr = np.asarray(p)
            if arr.ndim == 2 and arr.shape[1] >= 2:
                scores.append(float(arr[0, 1]))
            elif arr.ndim == 2 and arr.shape[1] == 1:
                scores.append(float(arr[0, 0]))
            else:
                scores.append(float(arr.ravel()[0]))
        return np.array(scores, dtype=np.float64)

    arr = np.asarray(proba)
    if arr.ndim == 2 and arr.shape[1] == n_labels:
        return arr[0].astype(np.float64, copy=False)
    if arr.ndim == 2 and arr.shape[1] == 2 and n_labels == 1:
        return np.array([float(arr[0, 1])], dtype=np.float64)
    raise ValueError(f"Unsupported predict_proba shape: {getattr(arr, 'shape', None)}")


class ToxicInferenceService:
    """Loads a serialized sklearn Pipeline (or compatible) and runs predict_proba."""

    def __init__(self, model_path: Path) -> None:
        self._model_path = model_path
        self._model: Any | None = None

    @property
    def labels(self) -> tuple[str, ...]:
        return LABELS

    def load(self) -> None:
        if not self._model_path.is_file():
            raise FileNotFoundError(
                f"Model file not found: {self._model_path}. "
                "Train in ml/ and export a pickle, or mount ./models in Docker."
            )
        repo_root = str(_REPO_ROOT)
        if repo_root not in sys.path:
            sys.path.insert(0, repo_root)
        with self._model_path.open("rb") as f:
            self._model = pickle.load(f)

    def ensure_loaded(self) -> None:
        if self._model is None:
            self.load()

    def predict_proba(self, text: str) -> dict[str, float]:
        self.ensure_loaded()
        assert self._model is not None
        raw = self._model.predict_proba([text])
        scores = _scores_from_predict_proba(raw, len(LABELS))
        if scores.shape[0] != len(LABELS):
            raise ValueError(
                f"Model output size {scores.shape[0]} does not match LABELS ({len(LABELS)})."
            )
        return {label: float(score) for label, score in zip(LABELS, scores, strict=True)}
