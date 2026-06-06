# Backend API Documentation

FastAPI service for **inference** of multi-label toxic comment classifiers. Contains no training logic — only artifact loading and HTTP API.

## Architecture

```text
backend/app/
├── main.py                 # Entry point, CORS, lifespan (model loading)
├── core/config.py          # Artifact paths, CORS, language detection threshold
├── api/routes.py           # REST endpoints, PCA + anchor maps, metrics
└── services/
    ├── registry.py         # EN/PL model registry (TF-IDF + BERT)
    ├── inference.py        # Sklearn pipeline → predict_proba
    ├── bert_inference.py   # Hugging Face BERT/HerBERT → sigmoid
    ├── projection.py       # Offline PCA validation cloud + live user projection
    ├── anchor_projection.py # Heuristic reference-anchor 2D map
    └── lang_detect.py      # gcld3 / langdetect → EN/PL routing
```

### Application lifecycle

1. On startup, `lifespan` creates `InferenceRegistry` and calls `load_all()`.
2. Each model loads in try/except: a missing artifact does not crash the server; it is marked `loaded: false`.
3. `/api/predict` requests go through the registry by `model` + `lang`.

### Supported models

| `model`     | EN artifact          | PL artifact           |
|-------------|----------------------|------------------------|
| `tfidf_lr`  | `models/model.pkl`   | `models/model_pl.pkl`  |
| `bert`      | `models/bert/`       | `models/bert_pl/`      |
| `both`      | both at once         | both at once           |

## Configuration (environment variables)

| Variable | Default | Description |
|----------|---------|-------------|
| `MODEL_PATH` | `{repo}/models/model.pkl` | Sklearn pipeline pickle (EN) |
| `BERT_MODEL_DIR` | `{repo}/models/bert` | Hugging Face directory (EN BERT) |
| `MODEL_PATH_PL` | `{repo}/models/model_pl.pkl` | Sklearn pipeline pickle (PL) |
| `BERT_MODEL_DIR_PL` | `{repo}/models/bert_pl` | Hugging Face directory (PL HerBERT) |
| `ML_EXPERIMENTS_DIR` | `{repo}/ml/experiments` | Metrics JSON for `/api/metrics` |
| `PROJECTIONS_DIR` | `{repo}/models/projections` | PCA scatter artifacts (`corpus.json`, `reducer.joblib`) |
| `LANG_DETECT_MIN_CONFIDENCE` | `0.70` | Minimum gcld3/langdetect confidence |

In Docker Compose, paths are overridden to `/models/...`, `/ml/experiments`, and `/models/projections`.

## API

### `GET /api/health`

Liveness check. Always returns `{"status": "ok"}`.

### `GET /api/ready`

Model readiness:

```json
{
  "model_loaded_en": true,
  "model_loaded_pl": true,
  "models_en": {"tfidf_lr": true, "bert": true},
  "models_pl": {"tfidf_lr": true, "bert": true},
  "tfidf_path_en": "...",
  "bert_dir_en": "..."
}
```

### `GET /api/models?lang=en|pl`

List of models with `loaded` flag and artifact path.

### `POST /api/detect-lang`

Body: `{"text": "..."}`

Response: `analysis_lang`, `confidence`, `is_reliable`, `source` (`gcld3` | `langdetect` | `fallback`).

### `POST /api/predict`

Body:

```json
{
  "text": "comment to analyze",
  "model": "tfidf_lr | bert | both",
  "lang": "auto | en | pl",
  "include_pca": false
}
```

- **`lang: auto`** — language detected via `lang_detect` (gcld3 in Docker, langdetect fallback on Windows).
- **`lang: en|pl`** — forced selection; detector is skipped (`lang_source: forced`).
- **`include_pca: false`** (default) — skip PCA validation scatter (no extra embedding pass). Reference anchor map is still returned. Set `true` for full PCA 2D/3D maps.

Response (main fields):

| Field | Description |
|-------|-------------|
| `probabilities` | Per-label probabilities (primary model; BERT when `both`) |
| `labels` | Label order for the selected language |
| `analysis_lang` | Language actually used (`en` / `pl`) |
| `requested_lang` | Value from the request |
| `lang_confidence` | Detector confidence (if not forced) |
| `pca_included` | Whether PCA maps were computed (`include_pca` request flag) |
| `similarity_projection` | PCA validation cloud + user point (empty when `include_pca: false`) |
| `reference_projection` | Heuristic anchor map (always when inference succeeds) |
| `projection_method` | e.g. `TruncatedSVD(50)+PCA(3)` or `PCA(3)` (PCA only) |
| `explained_variance_ratio` | Per principal component (PCA only) |
| `is_dual` | `true` when `model: both` |
| `probabilities_tfidf` / `probabilities_bert` | Separate scores in dual mode |
| `similarity_projection_tfidf` / `_bert` | PCA maps in dual mode (if `include_pca`) |
| `reference_projection_tfidf` / `_bert` | Anchor maps in dual mode |

### `GET /api/projection/corpus`

Validation-set scatter data (built offline):

| Query | Description |
|-------|-------------|
| `lang` | `en` \| `pl` |
| `model` | `tfidf_lr` \| `bert` |
| `error_filter` | `all` \| `correct` \| `errors` \| `false_positive` \| `false_negative` \| `label_mismatch` |

Each point includes `x`, `y`, `z`, `ground_truth_labels`, `predicted_labels`, `error_type`.

Build artifacts: `python -m ml.training.build_projection_maps` (see `docs/ML.md`).

### `GET /api/metrics`

Aggregated training metrics from `ml/experiments/*/metrics.json`. Built-in fallback values if files are missing.

## Label schemas

**English (Jigsaw):** `toxic`, `severe_toxic`, `obscene`, `threat`, `insult`, `identity_hate` — any active label means a violation.

**Polish (BAN-PL):** `safe`, `hate_speech`, `violence`, `vulgarity` — high `safe` means a safe comment.

Thresholds for "active" labels in projection come from `models/thresholds.json` (see `ml/labels.py`).

## Language detection

`app/services/lang_detect.py`:

1. Tries **gcld3** (requires Docker build deps: `g++`, `protobuf-compiler`).
2. Falls back to **langdetect** (pure Python, for local development).
3. Only `en` and `pl` are supported; unknown language → fallback to `en`.

## Running locally

```bash
cd backend
pip install -r requirements.txt
python -m uvicorn app.main:app --reload --host 127.0.0.1 --port 8010
```

Port **8010** is the default for the Vite proxy (`frontend/vite.config.ts`). Docker uses **8000**.

## Docker

```bash
docker compose up --build
```

Backend mounts `./models` (including `projections/`), `./ml`, and `./backend/app` (hot reload).
