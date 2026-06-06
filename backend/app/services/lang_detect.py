"""Language detection for routing EN/PL models.

Primary backend: Google Compact Language Detector 3 (gcld3) when installed
(Docker / Linux). Falls back to langdetect on platforms where gcld3 cannot
be built (e.g. Windows without protoc).
"""

from __future__ import annotations

from dataclasses import dataclass
from functools import lru_cache

from app.core.config import settings

SUPPORTED_LANGS = frozenset({"en", "pl"})


@dataclass(frozen=True)
class LangDetectResult:
    lang: str
    confidence: float
    is_reliable: bool
    source: str
    detected_code: str | None = None


@lru_cache(maxsize=1)
def _gcld3_detector():
    import gcld3

    return gcld3.NNetLanguageIdentifier(min_num_bytes=0, max_num_bytes=1000)


def _detect_with_gcld3(text: str, min_confidence: float) -> LangDetectResult | None:
    try:
        detector = _gcld3_detector()
    except ImportError:
        return None

    results = detector.FindTopNMostFreqLangs(text, num_langs=3)
    if not results:
        return None

    for candidate in results:
        code = candidate.language
        if code not in SUPPORTED_LANGS:
            continue
        reliable = candidate.is_reliable and candidate.probability >= min_confidence
        return LangDetectResult(
            lang=code,
            confidence=float(candidate.probability),
            is_reliable=reliable,
            source="gcld3",
            detected_code=code,
        )

    top = results[0]
    return LangDetectResult(
        lang="en",
        confidence=float(top.probability),
        is_reliable=False,
        source="fallback",
        detected_code=top.language,
    )


def _detect_with_langdetect(text: str, min_confidence: float) -> LangDetectResult:
    from langdetect import DetectorFactory, LangDetectException, detect_langs

    DetectorFactory.seed = 0
    try:
        candidates = detect_langs(text)
    except LangDetectException:
        return LangDetectResult("en", 0.0, False, "fallback")

    for candidate in candidates:
        if candidate.lang in SUPPORTED_LANGS:
            return LangDetectResult(
                lang=candidate.lang,
                confidence=float(candidate.prob),
                is_reliable=float(candidate.prob) >= min_confidence,
                source="langdetect",
                detected_code=candidate.lang,
            )

    top = candidates[0] if candidates else None
    return LangDetectResult(
        lang="en",
        confidence=float(top.prob) if top else 0.0,
        is_reliable=False,
        source="fallback",
        detected_code=top.lang if top else None,
    )


def detect_language(text: str, min_confidence: float | None = None) -> LangDetectResult:
    threshold = settings.lang_detect_min_confidence if min_confidence is None else min_confidence
    stripped = text.strip()
    if not stripped:
        return LangDetectResult("en", 0.0, False, "fallback")

    gcld3_result = _detect_with_gcld3(stripped, threshold)
    if gcld3_result is not None:
        return gcld3_result

    return _detect_with_langdetect(stripped, threshold)


def resolve_analysis_lang(text: str, requested: str) -> tuple[str, LangDetectResult]:
    """
    Resolve analysis language for /api/predict.

    - en/pl → forced, skip detection;
    - auto  → detect_language(text);
    - other → fallback to en.
    """
    req = (requested or "auto").lower()
    if req in SUPPORTED_LANGS:
        return req, LangDetectResult(req, 1.0, True, "forced", req)
    if req == "auto":
        detected = detect_language(text)
        return detected.lang, detected
    return "en", LangDetectResult("en", 0.0, False, "default")
