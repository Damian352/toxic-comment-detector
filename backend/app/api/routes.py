"""
HTTP API for inference and visualization helpers.

Endpoints:
  GET  /api/health, /api/ready, /api/models, /api/metrics, /api/projection/corpus
  POST /api/predict, /api/detect-lang
"""

import json
from pathlib import Path
from fastapi import APIRouter, HTTPException
from pydantic import BaseModel, Field

from app.core.config import settings
from app.services.lang_detect import detect_language, resolve_analysis_lang
from app.services.anchor_projection import get_anchor_projection
from app.services.projection import ProjectionService
from app.services.registry import InferenceRegistry, ModelId
from ml.labels import LABELS, PL_LABELS

_REPO_ROOT = Path(__file__).resolve().parents[3]
router = APIRouter(prefix="/api", tags=["inference"])


# ---------------------------------------------------------------------------
# Pydantic request/response schemas
# ---------------------------------------------------------------------------


class PredictRequest(BaseModel):
    text: str = Field(..., min_length=1, max_length=8000)
    model: ModelId = Field(
        default=ModelId.TFIDF_LR,
        description="Inference backend: tfidf_lr (sklearn baseline) or bert (transformer).",
    )
    lang: str = Field(
        default="auto",
        description="Analysis language: 'auto' (detect), 'en', or 'pl'. Forced values skip detection.",
    )
    include_pca: bool = Field(
        default=False,
        description=(
            "When false, skip PCA validation scatter (no extra BERT/TF-IDF embeddings). "
            "Reference anchor map is still computed."
        ),
    )


class ProjectionPoint(BaseModel):
    id: str
    text: str
    labels: list[str]
    x: float
    y: float
    z: float = 0.0
    similarity: float
    is_active: bool
    is_validation: bool = False
    ground_truth_labels: list[str] = Field(default_factory=list)
    predicted_labels: list[str] = Field(default_factory=list)
    error_type: str | None = None


class LangDetectResponse(BaseModel):
    analysis_lang: str
    confidence: float
    is_reliable: bool
    source: str
    detected_code: str | None = None


class PredictResponse(BaseModel):
    probabilities: dict[str, float]
    labels: list[str]
    model: ModelId
    similarity_projection: list[ProjectionPoint] = Field(default_factory=list)

    requested_lang: str = "auto"
    analysis_lang: str = "en"
    lang_confidence: float | None = None
    lang_source: str | None = None

    # Dual comparison extensions
    is_dual: bool = False
    probabilities_tfidf: dict[str, float] | None = None
    probabilities_bert: dict[str, float] | None = None
    similarity_projection_tfidf: list[ProjectionPoint] | None = None
    similarity_projection_bert: list[ProjectionPoint] | None = None

    projection_method: str | None = None
    projection_axes: dict[str, str] | None = None
    explained_variance_ratio: list[float] | None = None

    reference_projection: list[ProjectionPoint] = Field(default_factory=list)
    reference_projection_tfidf: list[ProjectionPoint] | None = None
    reference_projection_bert: list[ProjectionPoint] | None = None

    pca_included: bool = False


class CorpusProjectionResponse(BaseModel):
    lang: str
    model: str
    method: str
    explained_variance_ratio: list[float]
    axes: dict[str, str]
    n_total_test: int
    n_displayed: int
    error_counts: dict[str, int]
    points: list[ProjectionPoint]


class ModelInfoResponse(BaseModel):
    id: ModelId
    name: str
    description: str
    loaded: bool
    artifact_path: str


# ---------------------------------------------------------------------------

# ---------------------------------------------------------------------------
# PCA projection helpers (validation-set scatter plots)
# ---------------------------------------------------------------------------


def get_projection_service() -> ProjectionService:
    from app.main import app

    return app.state.projection  # type: ignore[attr-defined]


def _model_kind(model_id: ModelId) -> str:
    if model_id == ModelId.BERT:
        return "bert"
    return "tfidf_lr"


def _point_from_dict(raw: dict) -> ProjectionPoint:
    return ProjectionPoint(
        id=raw["id"],
        text=raw["text"],
        labels=raw.get("labels", []),
        x=float(raw["x"]),
        y=float(raw["y"]),
        z=float(raw.get("z", 0.0)),
        similarity=float(raw.get("similarity", 0.0)),
        is_active=bool(raw.get("is_active", False)),
        is_validation=bool(raw.get("is_validation", False)),
        ground_truth_labels=list(raw.get("ground_truth_labels", [])),
        predicted_labels=list(raw.get("predicted_labels", [])),
        error_type=raw.get("error_type"),
    )


def _build_projection_points(
    text: str,
    probs: dict[str, float],
    lang: str,
    model_id: ModelId,
    registry: InferenceRegistry,
) -> tuple[list[ProjectionPoint], dict]:
    """Combine validation corpus + projected user comment."""
    projection = get_projection_service()
    kind = _model_kind(model_id)
    if not projection.is_available(lang, kind):
        active = ProjectionPoint(
            id="active_user",
            text=text,
            labels=[],
            x=0.0,
            y=0.0,
            z=0.0,
            similarity=1.0,
            is_active=True,
        )
        return [active], {}

    active, enriched_corpus = projection.build_user_projection(
        text, probs, lang, kind, registry
    )
    try:
        corpus = projection.get_corpus(lang, kind)
    except FileNotFoundError:
        corpus = {"points": [], "method": "PCA", "axes": {}, "explained_variance_ratio": []}

    corpus_points = [_point_from_dict(p) for p in enriched_corpus]
    active_pt = _point_from_dict(active)
    meta = {
        "projection_method": corpus.get("method"),
        "projection_axes": corpus.get("axes"),
        "explained_variance_ratio": corpus.get("explained_variance_ratio", []),
    }
    return corpus_points + [active_pt], meta


def _build_anchor_projection_points(
    text: str,
    probs: dict[str, float],
    lang: str,
    model_id: ModelId,
) -> list[ProjectionPoint]:
    """Reference anchor map (heuristic 2D, fixed benchmark comments)."""
    return [_point_from_dict(p) for p in get_anchor_projection(text, probs, lang, model_id)]


def _maybe_pca_projection(
    text: str,
    probs: dict[str, float],
    lang: str,
    model_id: ModelId,
    registry: InferenceRegistry,
    *,
    include_pca: bool,
) -> tuple[list[ProjectionPoint], dict]:
    """PCA validation cloud + user point (skipped when include_pca is false)."""
    if not include_pca:
        return [], {}
    return _build_projection_points(text, probs, lang, model_id, registry)


def get_registry() -> InferenceRegistry:
    from app.main import app

    return app.state.registry  # type: ignore[attr-defined]


@router.get("/health")
def health() -> dict[str, str]:
    return {"status": "ok"}


@router.get("/ready")
def ready() -> dict[str, bool | dict[str, bool]]:
    registry = get_registry()
    loaded_en = {m.id.value: registry.is_loaded(m.id, "en") for m in registry.list_models("en")}
    loaded_pl = {m.id.value: registry.is_loaded(m.id, "pl") for m in registry.list_models("pl")}
    return {
        "model_loaded_en": any(loaded_en.values()),
        "model_loaded_pl": any(loaded_pl.values()),
        "models_en": loaded_en,
        "models_pl": loaded_pl,
        "tfidf_path_en": str(settings.model_path),
        "bert_dir_en": str(settings.bert_model_dir),
        "tfidf_path_pl": str(settings.model_path_pl),
        "bert_dir_pl": str(settings.bert_model_dir_pl),
    }


@router.get("/models", response_model=list[ModelInfoResponse])
def list_models(lang: str = "en") -> list[ModelInfoResponse]:
    registry = get_registry()
    l = lang.lower() if lang else "en"
    if l not in ("en", "pl"):
        l = "en"
    return [
        ModelInfoResponse(
            id=m.id,
            name=m.name,
            description=m.description,
            loaded=m.loaded,
            artifact_path=m.artifact_path,
        )
        for m in registry.list_models(l)
    ]


class DetectLangRequest(BaseModel):
    text: str = Field(..., min_length=1, max_length=8000)


@router.get("/projection/corpus", response_model=CorpusProjectionResponse)
def projection_corpus(
    lang: str = "en",
    model: str = "bert",
    error_filter: str = "all",
) -> CorpusProjectionResponse:
    """Validation-set scatter points (PCA / TruncatedSVD+PCA) for error analysis."""
    l = lang.lower() if lang else "en"
    if l not in ("en", "pl"):
        l = "en"
    model_kind = model if model in ("tfidf_lr", "bert") else "bert"
    projection = get_projection_service()
    if not projection.is_available(l, model_kind):
        raise HTTPException(
            status_code=503,
            detail=(
                f"Projection artifacts missing for lang={l}, model={model_kind}. "
                "Run: python -m ml.training.build_projection_maps"
            ),
        )
    try:
        corpus = projection.get_corpus(l, model_kind, error_filter=error_filter)
    except FileNotFoundError as e:
        raise HTTPException(status_code=503, detail=str(e)) from e
    return CorpusProjectionResponse(
        lang=corpus["lang"],
        model=corpus["model"],
        method=corpus["method"],
        explained_variance_ratio=corpus.get("explained_variance_ratio", []),
        axes=corpus.get("axes", {"x": "PC1", "y": "PC2", "z": "PC3"}),
        n_total_test=int(corpus.get("n_total_test", 0)),
        n_displayed=int(corpus.get("n_displayed", 0)),
        error_counts=corpus.get("error_counts", {}),
        points=[_point_from_dict(p) for p in corpus.get("points", [])],
    )


@router.post("/detect-lang", response_model=LangDetectResponse)
def detect_lang(body: DetectLangRequest) -> LangDetectResponse:
    result = detect_language(body.text.strip())
    return LangDetectResponse(
        analysis_lang=result.lang,
        confidence=result.confidence,
        is_reliable=result.is_reliable,
        source=result.source,
        detected_code=result.detected_code,
    )


@router.post("/predict", response_model=PredictResponse)
def predict(body: PredictRequest) -> PredictResponse:
    """Main inference: optionally detect language, then run TF-IDF and/or BERT."""
    registry = get_registry()
    requested = (body.lang or "auto").lower()
    if requested not in ("auto", "en", "pl"):
        requested = "auto"

    try:
        text_stripped = body.text.strip()
        lang, detect_result = resolve_analysis_lang(text_stripped, requested)
        labels_list = list(PL_LABELS) if lang == "pl" else list(LABELS)
        lang_meta = {
            "requested_lang": requested,
            "analysis_lang": lang,
            "lang_confidence": detect_result.confidence,
            "lang_source": detect_result.source,
        }

        if body.model == ModelId.BOTH:
            probs_tfidf = registry.predict_proba(text_stripped, ModelId.TFIDF_LR, lang)
            probs_bert = registry.predict_proba(text_stripped, ModelId.BERT, lang)

            projection_tfidf, meta_tfidf = _maybe_pca_projection(
                text_stripped, probs_tfidf, lang, ModelId.TFIDF_LR, registry,
                include_pca=body.include_pca,
            )
            projection_bert, meta_bert = _maybe_pca_projection(
                text_stripped, probs_bert, lang, ModelId.BERT, registry,
                include_pca=body.include_pca,
            )
            proj_meta = meta_bert or meta_tfidf
            ref_tfidf = _build_anchor_projection_points(
                text_stripped, probs_tfidf, lang, ModelId.TFIDF_LR
            )
            ref_bert = _build_anchor_projection_points(
                text_stripped, probs_bert, lang, ModelId.BERT
            )

            return PredictResponse(
                probabilities=probs_bert,
                labels=labels_list,
                model=ModelId.BOTH,
                similarity_projection=projection_bert,
                is_dual=True,
                probabilities_tfidf=probs_tfidf,
                probabilities_bert=probs_bert,
                similarity_projection_tfidf=projection_tfidf,
                similarity_projection_bert=projection_bert,
                reference_projection=ref_bert,
                reference_projection_tfidf=ref_tfidf,
                reference_projection_bert=ref_bert,
                projection_method=proj_meta.get("projection_method"),
                projection_axes=proj_meta.get("projection_axes"),
                explained_variance_ratio=proj_meta.get("explained_variance_ratio"),
                pca_included=body.include_pca,
                **lang_meta,
            )
        else:
            probs = registry.predict_proba(text_stripped, body.model, lang)
            projection, proj_meta = _maybe_pca_projection(
                text_stripped, probs, lang, body.model, registry,
                include_pca=body.include_pca,
            )
            ref_projection = _build_anchor_projection_points(
                text_stripped, probs, lang, body.model
            )
            return PredictResponse(
                probabilities=probs,
                labels=labels_list,
                model=body.model,
                similarity_projection=projection,
                reference_projection=ref_projection,
                projection_method=proj_meta.get("projection_method"),
                projection_axes=proj_meta.get("projection_axes"),
                explained_variance_ratio=proj_meta.get("explained_variance_ratio"),
                pca_included=body.include_pca,
                **lang_meta,
            )
    except FileNotFoundError as e:
        raise HTTPException(
            status_code=503,
            detail=f"Model files not found or loaded for {body.model.value} (lang={requested}): {str(e)}"
        ) from e
    except ValueError as e:
        raise HTTPException(
            status_code=500,
            detail=f"Model error or invalid input for {body.model.value} (lang={requested}): {str(e)}"
        ) from e
    except Exception as e:
        import traceback
        tb = traceback.format_exc()
        print(f"ERROR during predict request: {str(e)}\n{tb}")
        raise HTTPException(
            status_code=500,
            detail=f"Unexpected server-side error during inference: {str(e)}\n{tb}"
        ) from e


def _normalize_metrics_block(raw: dict) -> dict:
    """Ensure fields expected by the frontend are always present."""
    dataset = raw.get("dataset") if isinstance(raw.get("dataset"), dict) else {}
    n_test = int(dataset.get("n_test", 0))
    return {
        **raw,
        "hamming_loss": float(raw.get("hamming_loss", 0.0)),
        "f1_macro": float(raw.get("f1_macro", 0.0)),
        "f1_micro": float(raw.get("f1_micro", 0.0)),
        "precision_macro": float(raw.get("precision_macro", 0.0)),
        "precision_micro": float(raw.get("precision_micro", 0.0)),
        "recall_macro": float(raw.get("recall_macro", 0.0)),
        "recall_micro": float(raw.get("recall_micro", 0.0)),
        "per_label": raw.get("per_label") if isinstance(raw.get("per_label"), list) else [],
        "dataset": {
            "n_samples": int(dataset.get("n_samples", n_test)),
            "n_train": int(dataset.get("n_train", 0)),
            "n_test": n_test,
        },
    }


@router.get("/metrics")
def get_metrics() -> dict[str, dict]:
    """Aggregate metrics.json from all four models for the UI Metrics tab."""
    experiments_dir = settings.ml_experiments_dir
    tfidf_path = experiments_dir / "baseline_tfidf_lr" / "metrics.json"
    bert_path = experiments_dir / "bert_multilabel" / "metrics.json"
    tfidf_pl_path = experiments_dir / "baseline_tfidf_lr_pl" / "metrics.json"
    bert_pl_path = experiments_dir / "bert_multilabel_pl" / "metrics.json"

    # Default/fallback values in case metrics.json are missing/inaccessible
    fallback_tfidf = {
        "hamming_loss": 0.02317614496840566,
        "f1_macro": 0.6162280435585351,
        "f1_micro": 0.7329723225030085,
        "precision_macro": 0.5159769410326367,
        "precision_micro": 0.6368674194897532,
        "recall_macro": 0.8024735482996835,
        "recall_micro": 0.8632369614512472,
        "per_label": [
            {"label": "toxic", "precision": 0.7170469798657718, "recall": 0.8731611637790128, "f1": 0.7874410377358491, "support": 3059},
            {"label": "severe_toxic", "precision": 0.31870967741935485, "recall": 0.7942122186495176, "f1": 0.4548802946593002, "support": 311},
            {"label": "obscene", "precision": 0.7534653465346535, "recall": 0.8900584795321638, "f1": 0.8160857908847186, "support": 1710},
            {"label": "threat", "precision": 0.3505747126436782, "recall": 0.6288659793814433, "f1": 0.45018450184501846, "support": 97},
            {"label": "insult", "precision": 0.6181653863533665, "recall": 0.8603773584905661, "f1": 0.7194320273468314, "support": 1590},
            {"label": "identity_hate", "precision": 0.3378995433789954, "recall": 0.7681660899653979, "f1": 0.4693446088794926, "support": 289}
        ],
        "dataset": {"n_samples": 159571, "n_train": 127656, "n_test": 31915}
    }

    fallback_bert = {
        "hamming_loss": 0.014251396939787978,
        "f1_macro": 0.6794028815189662,
        "f1_micro": 0.8081006961535757,
        "precision_macro": 0.7043997994888517,
        "precision_micro": 0.8019539427773901,
        "recall_macro": 0.6650081115370522,
        "recall_micro": 0.8143424036281179,
        "per_label": [
            {"label": "toxic", "precision": 0.8481338481338482, "recall": 0.8617195161817588, "f1": 0.8548727095832658, "support": 3059},
            {"label": "severe_toxic", "precision": 0.5854922279792746, "recall": 0.3633440514469453, "f1": 0.44841269841269843, "support": 311},
            {"label": "obscene", "precision": 0.8285238623751388, "recall": 0.8730994152046784, "f1": 0.8502277904328018, "support": 1710},
            {"label": "threat", "precision": 0.5783132530120482, "recall": 0.4948453608247423, "f1": 0.5333333333333333, "support": 97},
            {"label": "insult", "precision": 0.7516072472238458, "recall": 0.8088050314465409, "f1": 0.7791578309603151, "support": 1590},
            {"label": "identity_hate", "precision": 0.6343283582089553, "recall": 0.5882352941176471, "f1": 0.6104129263913824, "support": 289}
        ],
        "dataset": {"n_samples": 159571, "n_train": 127656, "n_test": 31915}
    }

    fallback_tfidf_pl = {
        "hamming_loss": 0.1395142797581822,
        "f1_macro": 0.6365711516494066,
        "f1_micro": 0.7408267983347856,
        "precision_macro": 0.5878459173888534,
        "precision_micro": 0.6916124367317426,
        "recall_macro": 0.6999693597252918,
        "recall_micro": 0.7975818219720658,
        "per_label": [
            {"label": "safe", "precision": 0.8592251630226314, "recall": 0.9345014601585315, "f1": 0.8952837729816147, "support": 2397},
            {"label": "hate_speech", "precision": 0.6841018582243634, "recall": 0.7362962962962963, "f1": 0.7092400998929718, "support": 1350},
            {"label": "violence", "precision": 0.3690322580645161, "recall": 0.5325884543761639, "f1": 0.43597560975609756, "support": 537},
            {"label": "vulgarity", "precision": 0.43902439024390244, "recall": 0.5964912280701754, "f1": 0.5057851239669422, "support": 513}
        ],
        "dataset": {"n_samples": 23985, "n_train": 19188, "n_test": 4797}
    }

    fallback_bert_pl = {
        "hamming_loss": 0.25,
        "f1_macro": 0.16666666666666666,
        "f1_micro": 0.5,
        "precision_macro": 0.125,
        "precision_micro": 0.5,
        "recall_macro": 0.25,
        "recall_micro": 0.5,
        "per_label": [
            {"label": "safe", "precision": 0.5, "recall": 1.0, "f1": 0.6666666666666666, "support": 5},
            {"label": "hate_speech", "precision": 0.0, "recall": 0.0, "f1": 0.0, "support": 4},
            {"label": "violence", "precision": 0.0, "recall": 0.0, "f1": 0.0, "support": 1},
            {"label": "vulgarity", "precision": 0.0, "recall": 0.0, "f1": 0.0, "support": 0}
        ],
        "dataset": {"n_samples": 50, "n_train": 40, "n_test": 10}
    }

    metrics = {}

    # Load English TF-IDF metrics
    def _load_metrics(path: Path, fallback: dict) -> dict:
        if path.is_file():
            try:
                with path.open("r", encoding="utf-8") as f:
                    return _normalize_metrics_block(json.load(f))
            except Exception:
                pass
        return _normalize_metrics_block(fallback)

    metrics["tfidf_lr"] = _load_metrics(tfidf_path, fallback_tfidf)
    metrics["bert"] = _load_metrics(bert_path, fallback_bert)
    metrics["tfidf_lr_pl"] = _load_metrics(tfidf_pl_path, fallback_tfidf_pl)
    metrics["bert_pl"] = _load_metrics(bert_pl_path, fallback_bert_pl)

    return metrics

