"""
Heuristic 2D reference map with fixed anchor comments (legacy visualization).

Coordinates are derived from model probabilities (toxicity profile), not PCA.
Similarity blends probability-profile distance (80%) and lexical Jaccard (20%).
"""

from __future__ import annotations

import math
import sys
from pathlib import Path
from typing import Any

_REPO_ROOT = Path(__file__).resolve().parents[3]
if str(_REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(_REPO_ROOT))

from ml.labels import LABELS, PL_LABELS
from app.services.registry import ModelId
from ml.labels import active_labels_from_probs, get_per_label_thresholds

ANCHORS: list[dict[str, Any]] = [
    {"id": "anchor_1", "text": "Hello! I think this is a great contribution. Thanks for editing!", "labels": ["safe"], "vector": [0.01, 0.0, 0.01, 0.0, 0.01, 0.0], "x": -80.0, "y": 0.0},
    {"id": "anchor_2", "text": "Please do not revert my edits without discussing them on the talk page first.", "labels": ["safe"], "vector": [0.03, 0.0, 0.01, 0.0, 0.02, 0.0], "x": -60.0, "y": -20.0},
    {"id": "anchor_3", "text": "Are you sure about this source? It seems a bit outdated, but we can verify.", "labels": ["safe"], "vector": [0.02, 0.0, 0.0, 0.0, 0.01, 0.0], "x": -50.0, "y": 20.0},
    {"id": "anchor_4", "text": "Thank you for the explanation. That makes a lot of sense now. Cheers!", "labels": ["safe"], "vector": [0.01, 0.0, 0.0, 0.0, 0.01, 0.0], "x": -85.0, "y": -10.0},
    {"id": "anchor_5", "text": "You are an absolute idiot. Why are you even allowed to edit here? Go away!", "labels": ["toxic", "insult"], "vector": [0.95, 0.05, 0.10, 0.01, 0.90, 0.05], "x": 40.0, "y": -40.0},
    {"id": "anchor_6", "text": "Get the hell out of here, you piece of trash! Nobody wants you here.", "labels": ["toxic", "obscene", "insult"], "vector": [0.98, 0.15, 0.95, 0.01, 0.92, 0.05], "x": 60.0, "y": -50.0},
    {"id": "anchor_7", "text": "I will kill you. I know where you live and I'm going to track you down.", "labels": ["toxic", "severe_toxic", "threat"], "vector": [0.99, 0.85, 0.70, 0.98, 0.80, 0.10], "x": 95.0, "y": 85.0},
    {"id": "anchor_8", "text": "You better watch your back, buddy. Someone is going to break your legs.", "labels": ["toxic", "threat"], "vector": [0.95, 0.40, 0.30, 0.92, 0.60, 0.05], "x": 80.0, "y": 70.0},
    {"id": "anchor_9", "text": "Stupid immigrants, you all should be sent back to where you came from!", "labels": ["toxic", "identity_hate", "insult"], "vector": [0.92, 0.15, 0.20, 0.10, 0.85, 0.90], "x": 75.0, "y": 45.0},
    {"id": "anchor_10", "text": "This is f***ing awesome, I love this obscene shit!", "labels": ["obscene"], "vector": [0.60, 0.05, 0.95, 0.01, 0.30, 0.05], "x": 20.0, "y": -80.0},
    {"id": "anchor_11", "text": "Your religion is a joke and all of your people are brainwashed fools.", "labels": ["toxic", "identity_hate", "insult"], "vector": [0.90, 0.10, 0.15, 0.05, 0.80, 0.85], "x": 70.0, "y": 35.0},
    {"id": "anchor_12", "text": "I hate you so much, you disgust me. You are a horrible human being.", "labels": ["toxic", "insult"], "vector": [0.88, 0.08, 0.12, 0.02, 0.82, 0.04], "x": 50.0, "y": -25.0},
    {"id": "anchor_13", "text": "Oh, brilliant edit. Did you use your last two remaining brain cells for that?", "labels": ["toxic", "insult"], "vector": [0.85, 0.02, 0.05, 0.01, 0.75, 0.02], "x": 30.0, "y": -15.0},
    {"id": "anchor_14", "text": "F*** off and leave this page alone, you stupid f***ing asshole.", "labels": ["toxic", "obscene", "insult"], "vector": [0.99, 0.20, 0.98, 0.01, 0.95, 0.02], "x": 65.0, "y": -75.0},
    {"id": "anchor_15", "text": "Women are completely incapable of understanding logic, they shouldn't edit wiki.", "labels": ["toxic", "identity_hate", "insult"], "vector": [0.88, 0.05, 0.10, 0.02, 0.78, 0.88], "x": 68.0, "y": 30.0},
    {"id": "anchor_16", "text": "I will hunt you down, slice your throat, and burn your house. Mark my words.", "labels": ["toxic", "severe_toxic", "threat"], "vector": [0.99, 0.95, 0.85, 0.99, 0.88, 0.05], "x": 98.0, "y": 95.0},
    {"id": "anchor_17", "text": "Hi there! I've added a few references to back up the statistics. Let me know what you think!", "labels": ["safe"], "vector": [0.01, 0.0, 0.0, 0.0, 0.01, 0.0], "x": -92.0, "y": 5.0},
    {"id": "anchor_18", "text": "Could you please explain the reasoning behind this section? I'm trying to understand the context.", "labels": ["safe"], "vector": [0.02, 0.0, 0.0, 0.0, 0.01, 0.0], "x": -45.0, "y": 35.0},
    {"id": "anchor_19", "text": "This is getting a bit frustrating. We've been over this three times already. Please read the guidelines.", "labels": ["safe"], "vector": [0.15, 0.01, 0.02, 0.0, 0.05, 0.01], "x": -20.0, "y": 10.0},
    {"id": "anchor_20", "text": "LOL HAHAHAHA HELLO WORLD WIKIPEDIA IS GAY NYAN CAT NYAN CAT NYAN CAT", "labels": ["toxic"], "vector": [0.55, 0.01, 0.12, 0.0, 0.10, 0.15], "x": 10.0, "y": -10.0},
]

PL_ANCHORS: list[dict[str, Any]] = [
    {"id": "pl_anchor_1", "text": "Dziękuję bardzo za pomoc! Świetna robota.", "labels": ["safe"], "vector": [0.99, 0.01, 0.01, 0.01], "x": -80.0, "y": 0.0},
    {"id": "pl_anchor_2", "text": "Moim zdaniem ten artykuł jest bardzo ciekawy i dobrze napisany.", "labels": ["safe"], "vector": [0.98, 0.01, 0.0, 0.01], "x": -60.0, "y": 20.0},
    {"id": "pl_anchor_3", "text": "Czy możemy sprawdzić to źródło jeszcze raz? Wydaje mi się trochę stare.", "labels": ["safe"], "vector": [0.95, 0.02, 0.01, 0.02], "x": -45.0, "y": -10.0},
    {"id": "pl_anchor_4", "text": "Ty kompletny idioto, zamknij się wreszcie i nie pisz bzdur.", "labels": ["hate_speech"], "vector": [0.05, 0.95, 0.10, 0.40], "x": 50.0, "y": -20.0},
    {"id": "pl_anchor_5", "text": "Wracaj skąd przyszedłeś, nie chcemy tu takich pasożytów!", "labels": ["hate_speech"], "vector": [0.02, 0.98, 0.20, 0.15], "x": 75.0, "y": 45.0},
    {"id": "pl_anchor_6", "text": "Twoja religia to żart, a wszyscy wy jesteście nienormalni.", "labels": ["hate_speech"], "vector": [0.04, 0.92, 0.15, 0.10], "x": 70.0, "y": 35.0},
    {"id": "pl_anchor_7", "text": "Znajdę cię i połamię ci nogi, pożałujesz tego.", "labels": ["violence"], "vector": [0.01, 0.40, 0.95, 0.10], "x": 80.0, "y": 70.0},
    {"id": "pl_anchor_8", "text": "Zabiję cię, wiem gdzie mieszkasz i cię dopadnę.", "labels": ["violence"], "vector": [0.01, 0.50, 0.99, 0.05], "x": 95.0, "y": 85.0},
    {"id": "pl_anchor_9", "text": "Co to za g***o? Co ty p***dolisz człowieku?!", "labels": ["vulgarity"], "vector": [0.10, 0.20, 0.05, 0.95], "x": 30.0, "y": -65.0},
    {"id": "pl_anchor_10", "text": "Wyp***dalaj stąd ty głupi ch***u!", "labels": ["vulgarity", "hate_speech"], "vector": [0.01, 0.85, 0.15, 0.99], "x": 65.0, "y": -75.0},
]


def get_anchor_projection(
    text: str,
    probs: dict[str, float],
    lang: str = "en",
    model_id: ModelId = ModelId.BERT,
) -> list[dict[str, Any]]:
    """Build anchor map points + active user comment."""
    labels = PL_LABELS if lang == "pl" else LABELS
    anchors = PL_ANCHORS if lang == "pl" else ANCHORS
    backend_model = "tfidf_lr" if model_id == ModelId.TFIDF_LR else "bert"
    thresholds = get_per_label_thresholds(lang, backend_model, labels)
    user_vector = [probs.get(label, 0.0) for label in labels]
    p_max = max(user_vector) if user_vector else 0.0

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
    points: list[dict[str, Any]] = []

    for anchor in anchors:
        anchor_vector = anchor["vector"]
        dist = math.sqrt(sum((u - a) ** 2 for u, a in zip(user_vector, anchor_vector, strict=True)))
        profile_sim = 1.0 - (dist / math.sqrt(len(labels)))
        anchor_words = set(w.strip(".,!?\"'()[]") for w in anchor["text"].lower().split() if len(w) >= 3)
        if user_words or anchor_words:
            jaccard = len(user_words.intersection(anchor_words)) / len(user_words.union(anchor_words))
        else:
            jaccard = 0.0
        similarity = 0.8 * profile_sim + 0.2 * jaccard
        points.append(
            {
                "id": anchor["id"],
                "text": anchor["text"],
                "labels": anchor["labels"],
                "x": float(anchor["x"]),
                "y": float(anchor["y"]),
                "z": 0.0,
                "similarity": float(similarity),
                "is_active": False,
                "is_validation": False,
                "ground_truth_labels": anchor["labels"],
                "predicted_labels": [],
                "error_type": None,
            }
        )

    user_labels = active_labels_from_probs(probs, thresholds, lang)
    points.append(
        {
            "id": "active_user",
            "text": text,
            "labels": user_labels,
            "x": float(user_x),
            "y": float(user_y),
            "z": 0.0,
            "similarity": 1.0,
            "is_active": True,
            "is_validation": False,
            "ground_truth_labels": [],
            "predicted_labels": user_labels,
            "error_type": None,
        }
    )
    return points
