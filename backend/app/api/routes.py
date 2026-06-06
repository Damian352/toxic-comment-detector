"""
HTTP API for inference and visualization helpers.

Endpoints:
  GET  /api/health, /api/ready, /api/models, /api/metrics
  POST /api/predict, /api/detect-lang

This module also contains:
  - anchor comments (ANCHORS / PL_ANCHORS) for 2D projection;
  - get_similarity_projection() — heuristic similarity map to reference points.
"""

import json
import math
from pathlib import Path
from fastapi import APIRouter, HTTPException
from pydantic import BaseModel, Field

from app.core.config import settings
from app.services.inference import LABELS, PL_LABELS
from app.services.lang_detect import detect_language, resolve_analysis_lang
from ml.labels import active_labels_from_probs, get_per_label_thresholds
from app.services.registry import InferenceRegistry, ModelId

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


class ProjectionPoint(BaseModel):
    id: str
    text: str
    labels: list[str]
    x: float
    y: float
    similarity: float
    is_active: bool


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


class ModelInfoResponse(BaseModel):
    id: ModelId
    name: str
    description: str
    loaded: bool
    artifact_path: str


# ---------------------------------------------------------------------------
# Reference comments for 2D projection (English Jigsaw context).
# vector — fixed probability profile; x/y — coordinates on the UI map.
# ---------------------------------------------------------------------------

ANCHORS = [
    {
        "id": "anchor_1",
        "text": "Hello! I think this is a great contribution. Thanks for editing!",
        "labels": ["safe"],
        "vector": [0.01, 0.0, 0.01, 0.0, 0.01, 0.0],
        "x": -80.0,
        "y": 0.0
    },
    {
        "id": "anchor_2",
        "text": "Please do not revert my edits without discussing them on the talk page first.",
        "labels": ["safe"],
        "vector": [0.03, 0.0, 0.01, 0.0, 0.02, 0.0],
        "x": -60.0,
        "y": -20.0
    },
    {
        "id": "anchor_3",
        "text": "Are you sure about this source? It seems a bit outdated, but we can verify.",
        "labels": ["safe"],
        "vector": [0.02, 0.0, 0.0, 0.0, 0.01, 0.0],
        "x": -50.0,
        "y": 20.0
    },
    {
        "id": "anchor_4",
        "text": "Thank you for the explanation. That makes a lot of sense now. Cheers!",
        "labels": ["safe"],
        "vector": [0.01, 0.0, 0.0, 0.0, 0.01, 0.0],
        "x": -85.0,
        "y": -10.0
    },
    {
        "id": "anchor_5",
        "text": "You are an absolute idiot. Why are you even allowed to edit here? Go away!",
        "labels": ["toxic", "insult"],
        "vector": [0.95, 0.05, 0.10, 0.01, 0.90, 0.05],
        "x": 40.0,
        "y": -40.0
    },
    {
        "id": "anchor_6",
        "text": "Get the hell out of here, you piece of trash! Nobody wants you here.",
        "labels": ["toxic", "obscene", "insult"],
        "vector": [0.98, 0.15, 0.95, 0.01, 0.92, 0.05],
        "x": 60.0,
        "y": -50.0
    },
    {
        "id": "anchor_7",
        "text": "I will kill you. I know where you live and I'm going to track you down.",
        "labels": ["toxic", "severe_toxic", "threat"],
        "vector": [0.99, 0.85, 0.70, 0.98, 0.80, 0.10],
        "x": 95.0,
        "y": 85.0
    },
    {
        "id": "anchor_8",
        "text": "You better watch your back, buddy. Someone is going to break your legs.",
        "labels": ["toxic", "threat"],
        "vector": [0.95, 0.40, 0.30, 0.92, 0.60, 0.05],
        "x": 80.0,
        "y": 70.0
    },
    {
        "id": "anchor_9",
        "text": "Stupid immigrants, you all should be sent back to where you came from!",
        "labels": ["toxic", "identity_hate", "insult"],
        "vector": [0.92, 0.15, 0.20, 0.10, 0.85, 0.90],
        "x": 75.0,
        "y": 45.0
    },
    {
        "id": "anchor_10",
        "text": "This is f***ing awesome, I love this obscene shit!",
        "labels": ["obscene"],
        "vector": [0.60, 0.05, 0.95, 0.01, 0.30, 0.05],
        "x": 20.0,
        "y": -80.0
    },
    {
        "id": "anchor_11",
        "text": "Your religion is a joke and all of your people are brainwashed fools.",
        "labels": ["toxic", "identity_hate", "insult"],
        "vector": [0.90, 0.10, 0.15, 0.05, 0.80, 0.85],
        "x": 70.0,
        "y": 35.0
    },
    {
        "id": "anchor_12",
        "text": "I hate you so much, you disgust me. You are a horrible human being.",
        "labels": ["toxic", "insult"],
        "vector": [0.88, 0.08, 0.12, 0.02, 0.82, 0.04],
        "x": 50.0,
        "y": -25.0
    },
    {
        "id": "anchor_13",
        "text": "Oh, brilliant edit. Did you use your last two remaining brain cells for that?",
        "labels": ["toxic", "insult"],
        "vector": [0.85, 0.02, 0.05, 0.01, 0.75, 0.02],
        "x": 30.0,
        "y": -15.0
    },
    {
        "id": "anchor_14",
        "text": "F*** off and leave this page alone, you stupid f***ing asshole.",
        "labels": ["toxic", "obscene", "insult"],
        "vector": [0.99, 0.20, 0.98, 0.01, 0.95, 0.02],
        "x": 65.0,
        "y": -75.0
    },
    {
        "id": "anchor_15",
        "text": "Women are completely incapable of understanding logic, they shouldn't edit wiki.",
        "labels": ["toxic", "identity_hate", "insult"],
        "vector": [0.88, 0.05, 0.10, 0.02, 0.78, 0.88],
        "x": 68.0,
        "y": 30.0
    },
    {
        "id": "anchor_16",
        "text": "I will hunt you down, slice your throat, and burn your house. Mark my words.",
        "labels": ["toxic", "severe_toxic", "threat"],
        "vector": [0.99, 0.95, 0.85, 0.99, 0.88, 0.05],
        "x": 98.0,
        "y": 95.0
    },
    {
        "id": "anchor_17",
        "text": "Hi there! I've added a few references to back up the statistics. Let me know what you think!",
        "labels": ["safe"],
        "vector": [0.01, 0.0, 0.0, 0.0, 0.01, 0.0],
        "x": -92.0,
        "y": 5.0
    },
    {
        "id": "anchor_18",
        "text": "Could you please explain the reasoning behind this section? I'm trying to understand the context.",
        "labels": ["safe"],
        "vector": [0.02, 0.0, 0.0, 0.0, 0.01, 0.0],
        "x": -45.0,
        "y": 35.0
    },
    {
        "id": "anchor_19",
        "text": "This is getting a bit frustrating. We've been over this three times already. Please read the guidelines.",
        "labels": ["safe"],
        "vector": [0.15, 0.01, 0.02, 0.0, 0.05, 0.01],
        "x": -20.0,
        "y": 10.0
    },
    {
        "id": "anchor_20",
        "text": "LOL HAHAHAHA HELLO WORLD WIKIPEDIA IS GAY NYAN CAT NYAN CAT NYAN CAT",
        "labels": ["toxic"],
        "vector": [0.55, 0.01, 0.12, 0.0, 0.10, 0.15],
        "x": 10.0,
        "y": -10.0
    }
]

# Anchors for Polish BAN-PL corpus (4 labels: safe + 3 violation types).
PL_ANCHORS = [
    {
        "id": "pl_anchor_1",
        "text": "Dziękuję bardzo za pomoc! Świetna robota.",
        "labels": ["safe"],
        "vector": [0.99, 0.01, 0.01, 0.01],
        "x": -80.0,
        "y": 0.0
    },
    {
        "id": "pl_anchor_2",
        "text": "Moim zdaniem ten artykuł jest bardzo ciekawy i dobrze napisany.",
        "labels": ["safe"],
        "vector": [0.98, 0.01, 0.0, 0.01],
        "x": -60.0,
        "y": 20.0
    },
    {
        "id": "pl_anchor_3",
        "text": "Czy możemy sprawdzić to źródło jeszcze raz? Wydaje mi się trochę stare.",
        "labels": ["safe"],
        "vector": [0.95, 0.02, 0.01, 0.02],
        "x": -45.0,
        "y": -10.0
    },
    {
        "id": "pl_anchor_4",
        "text": "Ty kompletny idioto, zamknij się wreszcie i nie pisz bzdur.",
        "labels": ["hate_speech"],
        "vector": [0.05, 0.95, 0.10, 0.40],
        "x": 50.0,
        "y": -20.0
    },
    {
        "id": "pl_anchor_5",
        "text": "Wracaj skąd przyszedłeś, nie chcemy tu takich pasożytów!",
        "labels": ["hate_speech"],
        "vector": [0.02, 0.98, 0.20, 0.15],
        "x": 75.0,
        "y": 45.0
    },
    {
        "id": "pl_anchor_6",
        "text": "Twoja religia to żart, a wszyscy wy jesteście nienormalni.",
        "labels": ["hate_speech"],
        "vector": [0.04, 0.92, 0.15, 0.10],
        "x": 70.0,
        "y": 35.0
    },
    {
        "id": "pl_anchor_7",
        "text": "Znajdę cię i połamię ci nogi, pożałujesz tego.",
        "labels": ["violence"],
        "vector": [0.01, 0.40, 0.95, 0.10],
        "x": 80.0,
        "y": 70.0
    },
    {
        "id": "pl_anchor_8",
        "text": "Zabiję cię, wiem gdzie mieszkasz i cię dopadnę.",
        "labels": ["violence"],
        "vector": [0.01, 0.50, 0.99, 0.05],
        "x": 95.0,
        "y": 85.0
    },
    {
        "id": "pl_anchor_9",
        "text": "Co to za g***o? Co ty p***dolisz człowieku?!",
        "labels": ["vulgarity"],
        "vector": [0.10, 0.20, 0.05, 0.95],
        "x": 30.0,
        "y": -65.0
    },
    {
        "id": "pl_anchor_10",
        "text": "Wyp***dalaj stąd ty głupi ch***u!",
        "labels": ["vulgarity", "hate_speech"],
        "vector": [0.01, 0.85, 0.15, 0.99],
        "x": 65.0,
        "y": -75.0
    }
]


def get_similarity_projection(
    text: str,
    probs: dict[str, float],
    lang: str = "en",
    model_id: ModelId = ModelId.BERT,
) -> list[ProjectionPoint]:
    """
    Build points for 2D visualization: anchors + the active user comment.

    Coordinates (x, y) are a heuristic from dominant probabilities, not UMAP/t-SNE.
    similarity — blend of probability profile (80%) and lexical Jaccard (20%).
    """
    labels = PL_LABELS if lang == "pl" else LABELS
    anchors = PL_ANCHORS if lang == "pl" else ANCHORS
    backend_model = "tfidf_lr" if model_id == ModelId.TFIDF_LR else "bert"
    thresholds = get_per_label_thresholds(lang, backend_model, labels)
    user_vector = [probs.get(l, 0.0) for l in labels]
    p_max = max(user_vector) if user_vector else 0.0
    
    # Active comment position on the plane (different geometry for PL vs EN)
    if lang == "pl":
        p_safe = probs.get("safe", 0.0)
        p_hate = probs.get("hate_speech", 0.0)
        p_violence = probs.get("violence", 0.0)
        p_vulgarity = probs.get("vulgarity", 0.0)
        
        if p_safe >= thresholds.get("safe", 0.5):
            user_x = -80.0 + ((1.0 - p_safe) * 40.0)
            user_y = 0.0
        else:
            user_x = -30.0 + (120.0 * max(p_hate, p_violence, p_vulgarity))
            w_total = p_hate + p_violence + p_vulgarity + 0.001
            user_y = (p_violence * 80.0 + p_hate * 40.0 + p_vulgarity * -70.0) / w_total
    else:
        if p_max < 0.15:
            user_x = -80.0 + (p_max * 50.0)
            user_y = 0.0
        else:
            user_x = -30.0 + (120.0 * p_max)
            w_threat = user_vector[3]
            w_hate = user_vector[5]
            w_obscene = user_vector[2]
            w_insult = user_vector[4]
            w_total = w_threat + w_hate + w_obscene + w_insult + 0.001
            user_y = (w_threat * 80.0 + w_hate * 40.0 + w_obscene * -70.0 + w_insult * -30.0) / w_total

    user_words = set(w.strip(".,!?\"'()[]") for w in text.lower().split() if len(w) >= 3)
    points = []
    
    # Process anchors
    for anchor in anchors:
        anchor_vector = anchor["vector"]
        
        # Profile similarity (Euclidean distance based)
        dist = math.sqrt(sum((u - a)**2 for u, a in zip(user_vector, anchor_vector)))
        profile_sim = 1.0 - (dist / math.sqrt(len(labels)))
        
        # Lexical similarity
        anchor_words = set(w.strip(".,!?\"'()[]") for w in anchor["text"].lower().split() if len(w) >= 3)
        if user_words or anchor_words:
            jaccard = len(user_words.intersection(anchor_words)) / len(user_words.union(anchor_words))
        else:
            jaccard = 0.0
            
        similarity = 0.8 * profile_sim + 0.2 * jaccard
        
        points.append(ProjectionPoint(
            id=anchor["id"],
            text=anchor["text"],
            labels=anchor["labels"],
            x=anchor["x"],
            y=anchor["y"],
            similarity=float(similarity),
            is_active=False
        ))
        
    user_labels = active_labels_from_probs(probs, thresholds, lang)
        
    points.append(ProjectionPoint(
        id="active_user",
        text=text,
        labels=user_labels,
        x=user_x,
        y=user_y,
        similarity=1.0,
        is_active=True
    ))
    
    return points



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

            projection_tfidf = get_similarity_projection(
                text_stripped, probs_tfidf, lang, ModelId.TFIDF_LR
            )
            projection_bert = get_similarity_projection(
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
                **lang_meta,
            )
        else:
            probs = registry.predict_proba(text_stripped, body.model, lang)
            projection = get_similarity_projection(text_stripped, probs, lang, body.model)
            return PredictResponse(
                probabilities=probs,
                labels=labels_list,
                model=body.model,
                similarity_projection=projection,
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

