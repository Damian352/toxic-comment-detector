# Backend API Documentation

FastAPI service for **inference** of multi-label toxic comment classifiers. Contains no training logic — only artifact loading and HTTP API.

## Architecture

```text
backend/app/
├── main.py                 # Entry point, CORS, lifespan (model loading)
├── core/config.py          # Artifact paths, CORS, language detection threshold
├── api/routes.py           # REST endpoints, 2D projection, metrics
└── services/
    ├── registry.py         # EN/PL model registry (TF-IDF + BERT)
    ├── inference.py        # Sklearn pipeline → predict_proba
    ├── bert_inference.py   # Hugging Face BERT/HerBERT → sigmoid
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
| `LANG_DETECT_MIN_CONFIDENCE` | `0.70` | Minimum gcld3/langdetect confidence |

In Docker Compose, paths are overridden to `/models/...` and `/ml/experiments`.

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
  "lang": "auto | en | pl"
}
```

- **`lang: auto`** — language detected via `lang_detect` (gcld3 in Docker, langdetect fallback on Windows).
- **`lang: en|pl`** — forced selection; detector is skipped (`lang_source: forced`).

Response (main fields):

| Field | Description |
|-------|-------------|
| `probabilities` | Per-label probabilities (primary model; BERT when `both`) |
| `labels` | Label order for the selected language |
| `analysis_lang` | Language actually used (`en` / `pl`) |
| `requested_lang` | Value from the request |
| `lang_confidence` | Detector confidence (if not forced) |
| `similarity_projection` | Points for 2D visualization (anchors + active comment) |
| `is_dual` | `true` when `model: both` |
| `probabilities_tfidf` / `probabilities_bert` | Separate scores in dual mode |

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

Backend mounts `./models`, `./ml`, and `./backend/app` (hot reload).
