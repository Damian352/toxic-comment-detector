"""Shared multi-label taxonomy (Jigsaw Toxic Comment Classification)."""

LABELS: tuple[str, ...] = (
    "toxic",
    "severe_toxic",
    "obscene",
    "threat",
    "insult",
    "identity_hate",
)

DEFAULT_THRESHOLD = 0.5
