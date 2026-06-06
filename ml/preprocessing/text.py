"""Lightweight text preprocessing for toxic comment classification."""

from __future__ import annotations

import html
import re
from typing import Iterable

URL_PATTERN = re.compile(r"https?://\S+|www\.\S+", re.IGNORECASE)
HTML_TAG_PATTERN = re.compile(r"<[^>]+>")
WHITESPACE_PATTERN = re.compile(r"\s+")


def preprocess_text(text: str) -> str:
    """
    Normalize comment text before vectorization.

    Applied steps: HTML unescape, lowercase, URL removal, HTML tag removal,
    whitespace normalization. Punctuation, stopwords, and word forms are kept.
    """
    if not isinstance(text, str):
        return ""
    text = html.unescape(text)
    text = text.lower()
    text = URL_PATTERN.sub(" ", text)
    text = HTML_TAG_PATTERN.sub(" ", text)
    return WHITESPACE_PATTERN.sub(" ", text).strip()


def preprocess_text_pl(text: str) -> str:
    """
    Normalize Polish comment text before vectorization or transformer input.
    """
    if not isinstance(text, str):
        return ""
    text = text.replace("\n", " ")
    text = re.sub(r"{USERNAME}", "", text)
    text = re.sub(r"{URL}", "", text)
    text = re.sub(r"&gt;", "", text)
    text = re.sub(r"http\S+", "", text)
    text = re.sub(r"\s+", " ", text)
    return text.strip()


def preprocess_batch(texts: Iterable[str]) -> list[str]:
    """Apply ``preprocess_text`` to an iterable of comments (sklearn-compatible)."""
    return [preprocess_text(text) for text in texts]


def preprocess_batch_pl(texts: Iterable[str]) -> list[str]:
    """Apply ``preprocess_text_pl`` to an iterable of comments (sklearn-compatible)."""
    return [preprocess_text_pl(text) for text in texts]
