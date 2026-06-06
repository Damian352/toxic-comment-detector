"""Shared multi-label taxonomy (Jigsaw and Polish BAN-PL)."""

from __future__ import annotations

import json
from pathlib import Path

LABELS: tuple[str, ...] = (
    "toxic",
    "severe_toxic",
    "obscene",
    "threat",
    "insult",
    "identity_hate",
)

PL_LABELS: tuple[str, ...] = (
    "safe",
    "hate_speech",
    "violence",
    "vulgarity",
)

DEFAULT_THRESHOLD = 0.5

_REPO_ROOT = Path(__file__).resolve().parents[1]


def _registry_path() -> Path:
    return _REPO_ROOT / "models" / "thresholds.json"


def load_threshold_registry() -> dict:
    """Load consolidated per-model per-label thresholds if present."""
    path = _registry_path()
    if not path.is_file():
        return {}
    return json.loads(path.read_text(encoding="utf-8"))


def get_per_label_thresholds(
    lang: str,
    model_id: str,
    labels: tuple[str, ...] | list[str],
) -> dict[str, float]:
    """
    Return per-label thresholds for a language/model pair.

    Falls back to uniform DEFAULT_THRESHOLD when no tuned values exist.
    model_id: 'tfidf_lr' or 'bert'
    """
    registry = load_threshold_registry()
    lang_block = registry.get(lang, {})
    per_label = lang_block.get(model_id)
    if isinstance(per_label, dict) and per_label:
        return {label: float(per_label.get(label, DEFAULT_THRESHOLD)) for label in labels}
    return {label: DEFAULT_THRESHOLD for label in labels}


def is_label_active(label: str, probability: float, threshold: float, lang: str) -> bool:
    """Whether a label is considered active given language-specific semantics."""
    if lang == "pl" and label == "safe":
        return probability >= threshold
    if lang == "pl":
        return probability >= threshold
    return probability >= threshold


def active_labels_from_probs(
    probs: dict[str, float],
    thresholds: dict[str, float],
    lang: str,
) -> list[str]:
    """Derive human-readable active labels using per-label thresholds."""
    if lang == "pl":
        active = [l for l, p in probs.items() if l != "safe" and p >= thresholds.get(l, DEFAULT_THRESHOLD)]
        if not active and probs.get("safe", 0.0) >= thresholds.get("safe", DEFAULT_THRESHOLD):
            return ["safe"]
        if not active:
            return ["safe"]
        return active

    active = [l for l, p in probs.items() if p >= thresholds.get(l, DEFAULT_THRESHOLD)]
    if not active:
        return ["safe"]
    return active
