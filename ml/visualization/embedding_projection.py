"""
Build PCA-based 2D/3D projections of validation-set comments for scatter-plot analysis.

Pipeline:
  1. Reproduce the hold-out test split (same random_state as training).
  2. Extract embeddings (TF-IDF sparse features or BERT [CLS] vectors).
  3. Reduce dimensionality: TruncatedSVD + PCA for TF-IDF, PCA for BERT.
  4. Tag each point with ground truth, prediction, and error type (FP/FN/etc.).
  5. Persist reducer + corpus JSON for the FastAPI projection service.
"""

from __future__ import annotations

import json
import pickle
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Literal

import numpy as np
from scipy import sparse
from sklearn.decomposition import PCA, TruncatedSVD
from sklearn.model_selection import train_test_split
from sklearn.pipeline import Pipeline

from ml.labels import LABELS, PL_LABELS, active_labels_from_probs, get_per_label_thresholds

ModelKind = Literal["tfidf_lr", "bert"]
LangKind = Literal["en", "pl"]

DEFAULT_RANDOM_STATE = 42
DEFAULT_TEST_SIZE = 0.2
DEFAULT_MAX_POINTS = 800
DEFAULT_MAX_EMBED = 2500
DEFAULT_SVD_COMPONENTS = 50
TEXT_PREVIEW_LEN = 120


@dataclass(frozen=True)
class TestSplit:
    texts: list[str]
    y_true: np.ndarray
    indices: np.ndarray


def _repo_root() -> Path:
    return Path(__file__).resolve().parents[2]


def _tiny_demo_corpus_pl() -> tuple[list[str], np.ndarray]:
    """Small Polish fallback when BAN-PL.csv is unavailable."""
    texts = [
        "Dziękuję bardzo za pomoc, świetna robota!",
        "Moim zdaniem artykuł jest ciekawy i dobrze napisany.",
        "Ty kompletny idioto, zamknij się wreszcie.",
        "Znajdę cię i połamię ci nogi.",
        "Co to za głupota? Przestań spamować forum.",
        "Czy możemy omówić to na stronie dyskusji?",
        "Wracaj skąd przyszedłeś, nie chcemy tu takich pasożytów!",
        "Wypad stąd, ty głupi człowieku.",
    ]
    y = np.array(
        [
            [1, 0, 0, 0],
            [1, 0, 0, 0],
            [0, 1, 0, 1],
            [0, 0, 1, 0],
            [0, 0, 0, 1],
            [1, 0, 0, 0],
            [0, 1, 0, 0],
            [0, 1, 0, 1],
        ],
        dtype=np.int32,
    )
    return texts, y


def _truncate(text: str, max_len: int = TEXT_PREVIEW_LEN) -> str:
    text = text.replace("\n", " ").strip()
    if len(text) <= max_len:
        return text
    return text[: max_len - 3] + "..."


def _labels_for_lang(lang: LangKind) -> tuple[str, ...]:
    return PL_LABELS if lang == "pl" else LABELS


def _is_truly_toxic(y_row: np.ndarray, lang: LangKind) -> bool:
    if lang == "pl":
        if y_row.shape[0] >= 4:
            return bool(y_row[1:].max() > 0)
        return bool(y_row.max() > 0)
    return bool(y_row.sum() > 0)


def _active_labels_from_binary(y_row: np.ndarray, labels: tuple[str, ...], lang: LangKind) -> list[str]:
    if lang == "pl":
        active = [labels[i] for i, v in enumerate(y_row) if i > 0 and v > 0]
        if not active:
            return ["safe"]
        return active
    active = [labels[i] for i, v in enumerate(y_row) if v > 0]
    if not active:
        return ["safe"]
    return active


def classify_error_type(
    y_true: np.ndarray,
    y_pred: np.ndarray,
    lang: LangKind,
) -> str:
    """Classify a validation point for scatter-plot coloring."""
    truly = _is_truly_toxic(y_true, lang)
    predicted = _is_truly_toxic(y_pred, lang)
    if truly and predicted and np.array_equal(y_true, y_pred):
        return "correct"
    if not truly and not predicted:
        return "correct"
    if predicted and not truly:
        return "false_positive"
    if truly and not predicted:
        return "false_negative"
    return "label_mismatch"


def load_test_split(
    lang: LangKind,
    *,
    data_path: Path | None = None,
    test_size: float = DEFAULT_TEST_SIZE,
    random_state: int = DEFAULT_RANDOM_STATE,
    demo: bool = False,
) -> TestSplit:
    """Load dataset and return the same hold-out test split used in training."""
    if lang == "en":
        from ml.training.train_baseline import _tiny_demo_corpus, load_jigsaw_dataset

        path = data_path or _repo_root() / "data" / "raw" / "train.csv"
        if demo or not path.is_file():
            texts, y = _tiny_demo_corpus()
        else:
            texts, y = load_jigsaw_dataset(path)
        stratify = y[:, 0]
    else:
        from ml.training.train_baseline_pl import load_polish_dataset

        path = data_path or _repo_root() / "BAN-PL_2" / "BAN-PL.csv"
        if demo or not path.is_file():
            texts, y = _tiny_demo_corpus_pl()
        else:
            texts, y = load_polish_dataset(path)
        stratify = y[:, 0] if y.shape[0] > 1 else None

    indices = np.arange(len(texts))
    split_kwargs: dict[str, Any] = {
        "test_size": test_size,
        "random_state": random_state,
    }
    if stratify is not None and len(np.unique(stratify)) > 1:
        split_kwargs["stratify"] = stratify
    _, test_idx = train_test_split(indices, **split_kwargs)
    test_idx = np.asarray(test_idx, dtype=np.int64)
    return TestSplit(
        texts=[texts[i] for i in test_idx],
        y_true=y[test_idx],
        indices=test_idx,
    )


def _positive_class_scores(proba_item: np.ndarray) -> np.ndarray:
    arr = np.asarray(proba_item)
    if arr.ndim == 2 and arr.shape[1] >= 2:
        return arr[:, 1]
    if arr.ndim == 2 and arr.shape[1] == 1:
        return arr[:, 0]
    return arr.ravel()


def _scores_matrix(proba: Any, n_labels: int) -> np.ndarray:
    arr = np.asarray(proba)
    if arr.ndim == 2 and arr.shape[1] == n_labels:
        return arr.astype(np.float64)
    if isinstance(proba, list):
        return np.column_stack([_positive_class_scores(p) for p in proba]).astype(np.float64)
    raise ValueError(f"Unsupported predict_proba shape: {getattr(arr, 'shape', None)}")


def predict_binary_matrix(
    model: Any,
    texts: list[str],
    labels: tuple[str, ...],
    lang: LangKind,
    model_kind: ModelKind,
) -> np.ndarray:
    """Run batch inference → binary multi-label matrix."""
    thresholds = get_per_label_thresholds(lang, model_kind, labels)
    thresh_vec = np.array([thresholds[l] for l in labels], dtype=np.float64)

    if model_kind == "tfidf_lr":
        proba = model.predict_proba(texts)
        scores = _scores_matrix(proba, len(labels))
    else:
        import torch
        from transformers import AutoModelForSequenceClassification, AutoTokenizer

        from ml.training.bert_multilabel import DEFAULT_MAX_LENGTH, logits_to_probabilities

        model_dir = Path(model) if isinstance(model, (str, Path)) else None
        if model_dir is not None:
            tokenizer = AutoTokenizer.from_pretrained(model_dir)
            clf = AutoModelForSequenceClassification.from_pretrained(model_dir)
            clf.eval()
            device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
            clf.to(device)
            max_length = DEFAULT_MAX_LENGTH
            meta_path = model_dir / "labels.json"
            if meta_path.is_file():
                meta = json.loads(meta_path.read_text(encoding="utf-8"))
                max_length = int(meta.get("max_length", max_length))
            batch_scores: list[np.ndarray] = []
            batch_size = 32
            for start in range(0, len(texts), batch_size):
                batch = texts[start : start + batch_size]
                enc = tokenizer(
                    batch,
                    truncation=True,
                    padding=True,
                    max_length=max_length,
                    return_tensors="pt",
                )
                enc = {k: v.to(device) for k, v in enc.items()}
                with torch.no_grad():
                    logits = clf(**enc).logits.cpu().numpy()
                batch_scores.append(logits_to_probabilities(logits))
            scores = np.vstack(batch_scores)
        else:
            scores_list = []
            for text in texts:
                raw = model.predict_proba([text])
                scores_list.append(_scores_matrix(raw, len(labels))[0])
            scores = np.vstack(scores_list)

    return (scores >= thresh_vec).astype(np.int32)


def extract_tfidf_embeddings(pipeline: Any, texts: list[str]) -> sparse.csr_matrix:
    """Transform texts through preprocess + TF-IDF feature union."""
    preprocessed = pipeline.named_steps["preprocess"].transform(texts)
    return pipeline.named_steps["features"].transform(preprocessed)


def extract_bert_cls_embeddings(model_dir: Path, texts: list[str], batch_size: int = 32) -> np.ndarray:
    """Extract [CLS] token embeddings from a fine-tuned Hugging Face classifier."""
    import torch
    from transformers import AutoModelForSequenceClassification, AutoTokenizer

    from ml.training.bert_multilabel import DEFAULT_MAX_LENGTH

    tokenizer = AutoTokenizer.from_pretrained(model_dir)
    model = AutoModelForSequenceClassification.from_pretrained(model_dir)
    model.eval()
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model.to(device)

    max_length = DEFAULT_MAX_LENGTH
    meta_path = model_dir / "labels.json"
    if meta_path.is_file():
        meta = json.loads(meta_path.read_text(encoding="utf-8"))
        max_length = int(meta.get("max_length", max_length))

    encoder = getattr(model, "bert", None) or getattr(model, "roberta", None) or getattr(model, "base_model", model)

    out: list[np.ndarray] = []
    for start in range(0, len(texts), batch_size):
        batch = texts[start : start + batch_size]
        enc = tokenizer(
            batch,
            truncation=True,
            padding=True,
            max_length=max_length,
            return_tensors="pt",
        )
        enc = {k: v.to(device) for k, v in enc.items()}
        with torch.no_grad():
            if hasattr(encoder, "embeddings"):
                hidden = encoder(**enc).last_hidden_state[:, 0, :].cpu().numpy()
            else:
                hidden = model(**enc, output_hidden_states=True).hidden_states[-1][:, 0, :].cpu().numpy()
        out.append(hidden.astype(np.float32))
    return np.vstack(out)


def fit_reducer(
    X: np.ndarray | sparse.csr_matrix,
    *,
    n_components: int = 3,
    svd_components: int = DEFAULT_SVD_COMPONENTS,
) -> tuple[Any, str]:
    """Fit TruncatedSVD+PCA (sparse) or PCA (dense). Returns (fitted, method_name)."""
    n_samples = X.shape[0]
    n_comp = min(n_components, n_samples)
    if n_comp < 2:
        n_comp = min(2, n_samples)

    if sparse.issparse(X):
        svd_dim = min(svd_components, X.shape[1] - 1, n_samples - 1)
        svd_dim = max(svd_dim, n_comp)
        reducer = Pipeline(
            [
                ("svd", TruncatedSVD(n_components=svd_dim, random_state=DEFAULT_RANDOM_STATE)),
                ("pca", PCA(n_components=n_comp, random_state=DEFAULT_RANDOM_STATE)),
            ]
        )
        method = f"TruncatedSVD({svd_dim})+PCA({n_comp})"
    else:
        dense_dim = min(n_comp, X.shape[1], n_samples)
        reducer = PCA(n_components=dense_dim, random_state=DEFAULT_RANDOM_STATE)
        method = f"PCA({dense_dim})"

    reducer.fit(X)
    return reducer, method


def _as_2d_samples(X: np.ndarray | sparse.csr_matrix) -> np.ndarray | sparse.csr_matrix:
    """Ensure sklearn reducers receive a 2D matrix (n_samples × n_features)."""
    if sparse.issparse(X):
        if X.shape[0] != 1 and X.ndim == 1:
            return X.reshape(1, -1)
        return X
    arr = np.asarray(X, dtype=np.float64)
    if arr.ndim == 1:
        return arr.reshape(1, -1)
    return arr


def transform_reducer(reducer: Any, X: np.ndarray | sparse.csr_matrix) -> np.ndarray:
    X = _as_2d_samples(X)
    if sparse.issparse(X) and isinstance(reducer, PCA):
        X = X.toarray()
    return reducer.transform(X)


def explained_variance_ratio(reducer: Any) -> list[float]:
    if isinstance(reducer, Pipeline):
        pca = reducer.named_steps["pca"]
        return [float(v) for v in pca.explained_variance_ratio_]
    return [float(v) for v in reducer.explained_variance_ratio_]


def _display_bounds(
    coords: np.ndarray,
    *,
    margin_ratio: float = 0.15,
) -> tuple[np.ndarray, np.ndarray]:
    """Raw PCA bounds with symmetric margin so live points are less likely to clip."""
    mins = coords.min(axis=0)
    maxs = coords.max(axis=0)
    span = maxs - mins
    span[span < 1e-9] = 1.0
    padded = span * (1.0 + margin_ratio)
    center = (mins + maxs) / 2.0
    return center - padded / 2.0, center + padded / 2.0


def normalize_to_display(
    coords: np.ndarray,
    scale: float = 90.0,
    *,
    margin_ratio: float = 0.15,
) -> tuple[np.ndarray, dict[str, list[float]]]:
    """Scale PCA coords to roughly [-scale, scale] for the UI."""
    mins, maxs = _display_bounds(coords, margin_ratio=margin_ratio)
    span = maxs - mins
    span[span < 1e-9] = 1.0
    normalized = (coords - mins) / span * (2 * scale) - scale
    meta = {
        "mins": [float(x) for x in mins],
        "maxs": [float(x) for x in maxs],
        "scale": scale,
        "margin_ratio": margin_ratio,
    }
    return normalized.astype(np.float32), meta


def stratified_sample_indices(
    error_types: list[str],
    max_points: int,
    rng: np.random.Generator,
) -> np.ndarray:
    """Sample validation points with emphasis on misclassifications."""
    n = len(error_types)
    if n <= max_points:
        return np.arange(n)

    errors = [i for i, e in enumerate(error_types) if e != "correct"]
    correct = [i for i, e in enumerate(error_types) if e == "correct"]

    error_quota = min(len(errors), max(max_points // 2, max_points - 100))
    correct_quota = max_points - error_quota

    pick_error = (
        rng.choice(errors, size=error_quota, replace=False) if error_quota and errors else np.array([], dtype=int)
    )
    remaining = correct_quota
    if len(pick_error) < error_quota and errors:
        remaining += error_quota - len(pick_error)

    pick_correct = (
        rng.choice(correct, size=min(remaining, len(correct)), replace=False)
        if correct
        else np.array([], dtype=int)
    )
    chosen = np.concatenate([pick_error, pick_correct])
    if len(chosen) < max_points:
        rest = np.setdiff1d(np.arange(n), chosen)
        extra = rng.choice(rest, size=min(max_points - len(chosen), len(rest)), replace=False)
        chosen = np.concatenate([chosen, extra])
    return np.sort(chosen)


def build_projection_bundle(
    *,
    lang: LangKind,
    model_kind: ModelKind,
    model_artifact: Path,
    out_dir: Path,
    data_path: Path | None = None,
    demo: bool = False,
    test_size: float = DEFAULT_TEST_SIZE,
    random_state: int = DEFAULT_RANDOM_STATE,
    max_points: int = DEFAULT_MAX_POINTS,
    max_embed: int = DEFAULT_MAX_EMBED,
    svd_components: int = DEFAULT_SVD_COMPONENTS,
) -> dict[str, Any]:
    """
    Build reducer.joblib + corpus.json for one (lang, model) pair.

    Returns manifest metadata dict.
    """
    labels = _labels_for_lang(lang)
    split = load_test_split(
        lang,
        data_path=data_path,
        test_size=test_size,
        random_state=random_state,
        demo=demo,
    )

    # Subsample before embedding when the hold-out set is large (BERT is expensive).
    rng = np.random.default_rng(random_state)
    work_idx = np.arange(len(split.texts))
    if len(work_idx) > max_embed:
        work_idx = np.sort(rng.choice(work_idx, size=max_embed, replace=False))
    work_texts = [split.texts[i] for i in work_idx]
    work_y_true = split.y_true[work_idx]
    work_orig_idx = split.indices[work_idx]

    if model_kind == "tfidf_lr":
        with model_artifact.open("rb") as f:
            clf = pickle.load(f)
        embeddings = extract_tfidf_embeddings(clf, work_texts)
        y_pred = predict_binary_matrix(clf, work_texts, labels, lang, "tfidf_lr")
    else:
        embeddings = extract_bert_cls_embeddings(model_artifact, work_texts)
        y_pred = predict_binary_matrix(model_artifact, work_texts, labels, lang, "bert")

    reducer, method = fit_reducer(embeddings, n_components=3, svd_components=svd_components)
    coords_3d = transform_reducer(reducer, embeddings)
    coords_display, norm_meta = normalize_to_display(coords_3d)

    error_types = [
        classify_error_type(work_y_true[i], y_pred[i], lang) for i in range(len(work_texts))
    ]
    sample_idx = stratified_sample_indices(error_types, max_points, rng)

    points: list[dict[str, Any]] = []
    for i in sample_idx:
        gt_labels = _active_labels_from_binary(work_y_true[i], labels, lang)
        pred_labels = _active_labels_from_binary(y_pred[i], labels, lang)
        points.append(
            {
                "id": f"val_{int(work_orig_idx[i])}",
                "text": _truncate(work_texts[i]),
                "x": float(coords_display[i, 0]),
                "y": float(coords_display[i, 1]),
                "z": float(coords_display[i, 2]) if coords_display.shape[1] > 2 else 0.0,
                "ground_truth_labels": gt_labels,
                "predicted_labels": pred_labels,
                "error_type": error_types[i],
                "is_active": False,
                "is_validation": True,
                "similarity": 0.0,
            }
        )

    out_dir.mkdir(parents=True, exist_ok=True)
    reducer_payload = {
        "reducer": reducer,
        "method": method,
        "lang": lang,
        "model": model_kind,
        "labels": list(labels),
        "norm": norm_meta,
        "explained_variance_ratio": explained_variance_ratio(reducer),
        "random_state": random_state,
        "test_size": test_size,
        "embedding": "tfidf" if model_kind == "tfidf_lr" else "bert_cls",
    }
    reducer_path = out_dir / "reducer.joblib"
    with reducer_path.open("wb") as f:
        pickle.dump(reducer_payload, f)

    corpus = {
        "lang": lang,
        "model": model_kind,
        "method": method,
        "explained_variance_ratio": explained_variance_ratio(reducer),
        "axes": {"x": "PC1", "y": "PC2", "z": "PC3"},
        "random_state": random_state,
        "test_size": test_size,
        "n_total_test": len(split.texts),
        "n_embedded": len(work_texts),
        "n_displayed": len(points),
        "error_counts": {
            k: sum(1 for e in error_types if e == k)
            for k in ("correct", "false_positive", "false_negative", "label_mismatch")
        },
        "points": points,
    }
    corpus_path = out_dir / "corpus.json"
    corpus_path.write_text(json.dumps(corpus, indent=2, ensure_ascii=False), encoding="utf-8")

    return {
        "lang": lang,
        "model": model_kind,
        "method": method,
        "reducer_path": str(reducer_path),
        "corpus_path": str(corpus_path),
        "n_total_test": len(split.texts),
        "n_displayed": len(points),
    }


def apply_display_norm(
    coords: np.ndarray,
    norm_meta: dict[str, Any],
    *,
    clip: bool = True,
) -> np.ndarray:
    """Map raw PCA coordinates to UI scale using corpus-fitted bounds."""
    mins = np.asarray(norm_meta["mins"], dtype=np.float64)
    maxs = np.asarray(norm_meta["maxs"], dtype=np.float64)
    scale = float(norm_meta.get("scale", 90.0))
    span = maxs - mins
    span[span < 1e-9] = 1.0
    if coords.ndim == 1:
        coords = coords.reshape(1, -1)
    out = (coords - mins) / span * (2 * scale) - scale
    if clip:
        out = np.clip(out, -scale, scale)
    return out.astype(np.float32)


def project_embedding(reducer_payload: dict[str, Any], embedding: np.ndarray | sparse.csr_matrix) -> np.ndarray:
    """Transform one embedding vector → display-scaled (x, y, z)."""
    reducer = reducer_payload["reducer"]
    raw = transform_reducer(reducer, embedding)
    if raw.ndim == 1:
        raw = raw.reshape(1, -1)
    return apply_display_norm(raw, reducer_payload["norm"])[0]


def cosine_similarity(a: np.ndarray, b: np.ndarray) -> float:
    a = np.asarray(a).ravel().astype(np.float64)
    b = np.asarray(b).ravel().astype(np.float64)
    denom = np.linalg.norm(a) * np.linalg.norm(b)
    if denom < 1e-12:
        return 0.0
    return float(np.dot(a, b) / denom)
