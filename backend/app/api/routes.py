import json
import math
from pathlib import Path
from fastapi import APIRouter, HTTPException
from pydantic import BaseModel, Field

from app.core.config import settings
from app.services.inference import LABELS
from app.services.registry import InferenceRegistry, ModelId

_REPO_ROOT = Path(__file__).resolve().parents[3]
router = APIRouter(prefix="/api", tags=["inference"])


class PredictRequest(BaseModel):
    text: str = Field(..., min_length=1, max_length=8000)
    model: ModelId = Field(
        default=ModelId.TFIDF_LR,
        description="Inference backend: tfidf_lr (sklearn baseline) or bert (transformer).",
    )


class ProjectionPoint(BaseModel):
    id: str
    text: str
    labels: list[str]
    x: float
    y: float
    similarity: float
    is_active: bool


class PredictResponse(BaseModel):
    probabilities: dict[str, float]
    labels: list[str] = Field(default_factory=lambda: list(LABELS))
    model: ModelId
    similarity_projection: list[ProjectionPoint] = Field(default_factory=list)
    
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


def get_similarity_projection(text: str, probs: dict[str, float]) -> list[ProjectionPoint]:
    user_vector = [probs.get(l, 0.0) for l in LABELS]
    p_max = max(user_vector)
    
    # Calculate user coords
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
    for anchor in ANCHORS:
        anchor_vector = anchor["vector"]
        
        # Profile similarity (Euclidean distance based)
        dist = math.sqrt(sum((u - a)**2 for u, a in zip(user_vector, anchor_vector)))
        profile_sim = 1.0 - (dist / math.sqrt(6))
        
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
        
    user_labels = [l for l in LABELS if probs.get(l, 0.0) >= 0.5]
    if not user_labels:
        user_labels = ["safe"]
        
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
    loaded = {m.id.value: m.loaded for m in registry.list_models()}
    return {
        "model_loaded": registry.any_loaded,
        "models": loaded,
        "tfidf_path": str(settings.model_path),
        "bert_dir": str(settings.bert_model_dir),
    }


@router.get("/models", response_model=list[ModelInfoResponse])
def list_models() -> list[ModelInfoResponse]:
    registry = get_registry()
    return [
        ModelInfoResponse(
            id=m.id,
            name=m.name,
            description=m.description,
            loaded=m.loaded,
            artifact_path=m.artifact_path,
        )
        for m in registry.list_models()
    ]


@router.post("/predict", response_model=PredictResponse)
def predict(body: PredictRequest) -> PredictResponse:
    registry = get_registry()
    try:
        text_stripped = body.text.strip()
        if body.model == ModelId.BOTH:
            # Run both models
            probs_tfidf = registry.predict_proba(text_stripped, ModelId.TFIDF_LR)
            probs_bert = registry.predict_proba(text_stripped, ModelId.BERT)
            
            projection_tfidf = get_similarity_projection(text_stripped, probs_tfidf)
            projection_bert = get_similarity_projection(text_stripped, probs_bert)
            
            # Return both with BERT as primary fallback in 'probabilities' and 'similarity_projection'
            return PredictResponse(
                probabilities=probs_bert,
                model=ModelId.BOTH,
                similarity_projection=projection_bert,
                is_dual=True,
                probabilities_tfidf=probs_tfidf,
                probabilities_bert=probs_bert,
                similarity_projection_tfidf=projection_tfidf,
                similarity_projection_bert=projection_bert
            )
        else:
            probs = registry.predict_proba(text_stripped, body.model)
            projection = get_similarity_projection(text_stripped, probs)
            return PredictResponse(probabilities=probs, model=body.model, similarity_projection=projection)
    except FileNotFoundError as e:
        raise HTTPException(status_code=503, detail=str(e)) from e
    except ValueError as e:
        raise HTTPException(status_code=500, detail=str(e)) from e


@router.get("/metrics")
def get_metrics() -> dict[str, dict]:
    # Paths to metrics
    tfidf_path = _REPO_ROOT / "ml" / "experiments" / "baseline_tfidf_lr" / "metrics.json"
    bert_path = _REPO_ROOT / "ml" / "experiments" / "bert_multilabel" / "metrics.json"

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

    metrics = {}

    # Load TF-IDF metrics
    if tfidf_path.is_file():
        try:
            with tfidf_path.open("r", encoding="utf-8") as f:
                metrics["tfidf_lr"] = json.load(f)
        except Exception:
            metrics["tfidf_lr"] = fallback_tfidf
    else:
        metrics["tfidf_lr"] = fallback_tfidf

    # Load BERT metrics
    if bert_path.is_file():
        try:
            with bert_path.open("r", encoding="utf-8") as f:
                metrics["bert"] = json.load(f)
        except Exception:
            metrics["bert"] = fallback_bert
    else:
        metrics["bert"] = fallback_bert

    return metrics

