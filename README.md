# Toxic Comment Detector

Multi-component project for **multi-label toxic comment classification** in English (Jigsaw) and Polish (BAN-PL).

| Component | Purpose |
|-----------|---------|
| `ml/` | Training, evaluation, threshold tuning, notebooks |
| `backend/` | FastAPI inference (TF-IDF + BERT/HerBERT) |
| `frontend/` | React UI: analysis, model comparison, metrics |
| `models/` | Artifacts (pickle, Hugging Face weights) |

**Detailed documentation:**

- [Backend API](docs/BACKEND.md) — endpoints, configuration, language detection
- [ML Pipeline](docs/ML.md) — training, metrics, thresholds, datasets

---

## Current capabilities

| Area | Description |
|------|-------------|
| **EN models** | TF-IDF+LR (`model.pkl`), BERT (`models/bert/`) |
| **PL models** | TF-IDF+LR (`model_pl.pkl`), HerBERT (`models/bert_pl/`) |
| **API** | `/api/predict` with `model`: `tfidf_lr` \| `bert` \| `both` |
| **Language** | `lang`: `auto` (gcld3/langdetect), `en`, `pl` |
| **Thresholds** | Per-label from `models/thresholds.json` (no retraining) |
| **UI** | Radar chart, 2D projection, dual mode, metrics tab |

### Labels

- **EN:** `toxic`, `severe_toxic`, `obscene`, `threat`, `insult`, `identity_hate`
- **PL:** `safe`, `hate_speech`, `violence`, `vulgarity`

---

## Screenshots

Demo captures from the React UI (`images/demo/`). Run the [quick start](#quick-start) below to reproduce them locally.

### Analysis (English)

**Safe comment — single-model view with auto language detection (BERT):**

![English analysis — safe comment](<images/demo/1-basic presentation-En.png>)

**Dual comparison mode — TF-IDF vs BERT on a threat comment:**

![English dual comparison — threat](<images/demo/2-dual comparison english.png>)

### Semantic space mapping

**2D vector alignment and nearest-neighbor references (dual mode, English):**

![Semantic space mapping — English overview](<images/demo/3-Semantic Space Mapping And Similarities 1.png>)

**Interactive tooltip on the active comment (TF-IDF projection):**

![Semantic space mapping — active comment tooltip](<images/demo/4-Semantic Space Mapping And Similarities 2.png>)

### Analysis (Polish)

**Dual comparison on Polish text — HerBERT vs TF-IDF (BAN-PL labels):**

![Polish dual comparison](<images/demo/5-dual comparison polish.png>)

**Semantic space mapping for the Polish corpus:**

![Semantic space mapping — Polish](<images/demo/6-Semantic Space Mapping And Similarities 3 Polish.png>)

### Model metrics dashboard

**Hold-out metrics: TF-IDF vs BERT on Jigsaw (EN):**

![Metrics dashboard — English](<images/demo/7-Metrics EN.png>)

**Hold-out metrics: TF-IDF vs HerBERT on BAN-PL (PL):**

![Metrics dashboard — Polish](<images/demo/8-Metrics PL.png>)

---

## Quick start

### 1. Training (from repository root)

```bash
pip install -r ml/requirements.txt

# English (requires data/raw/train.csv or --demo)
python -m ml.training.train_baseline
python -m ml.training.train_bert --demo --epochs 3

# Polish (requires BAN-PL_2/BAN-PL.csv)
python -m ml.training.train_baseline_pl
python -m ml.training.train_bert_pl --epochs 3

# Threshold tuning (optional)
python -m ml.training.tune_thresholds
```

### 2. Backend

```bash
cd backend
pip install -r requirements.txt
python -m uvicorn app.main:app --reload --host 127.0.0.1 --port 8010
```

Swagger: `http://127.0.0.1:8010/docs`

### 3. Frontend

```bash
cd frontend
npm install
npm run dev
```

UI: `http://127.0.0.1:5173` (proxies `/api` → port 8010)

### Docker Compose

```bash
docker compose up --build
```

Backend: `localhost:8000`, Frontend: `localhost:5173`

---

## API reference

Interactive OpenAPI docs: **`http://127.0.0.1:8010/docs`** (local) or **`http://localhost:8000/docs`** (Docker).

All routes are prefixed with `/api`. The frontend dev server proxies `/api/*` to the backend (see [Configuration](#configuration)).

### Overview

| Method | Path | Description |
|--------|------|-------------|
| GET | `/` | Service name and link to docs |
| GET | `/api/health` | Liveness probe |
| GET | `/api/ready` | Which model artifacts are loaded (EN/PL) |
| GET | `/api/models` | Available backends and artifact paths |
| GET | `/api/metrics` | Training/evaluation metrics for the UI dashboard |
| POST | `/api/detect-lang` | Language detection only (no inference) |
| POST | `/api/predict` | Multi-label toxicity inference |

### `GET /api/health`

Always returns HTTP 200 when the process is up:

```json
{ "status": "ok" }
```

Use for load balancers and container health checks.

### `GET /api/ready`

Reports whether each artifact was found and loaded at startup:

```json
{
  "model_loaded_en": true,
  "model_loaded_pl": true,
  "models_en": { "tfidf_lr": true, "bert": true },
  "models_pl": { "tfidf_lr": true, "bert": true },
  "tfidf_path_en": "/path/to/models/model.pkl",
  "bert_dir_en": "/path/to/models/bert",
  "tfidf_path_pl": "/path/to/models/model_pl.pkl",
  "bert_dir_pl": "/path/to/models/bert_pl"
}
```

Missing files do not crash the server; the corresponding `loaded` flag is `false` and `/api/predict` returns **503** for that model.

### `GET /api/models?lang=en|pl`

Query parameter `lang` selects which artifact set to describe (`en` default).

```json
[
  {
    "id": "tfidf_lr",
    "name": "TF-IDF + Logistic Regression",
    "description": "Sparse word/char n-grams with One-vs-Rest logistic regression (sklearn).",
    "loaded": true,
    "artifact_path": "models/model.pkl"
  },
  {
    "id": "bert",
    "name": "BERT",
    "description": "Fine-tuned transformer encoder with multi-label sigmoid head (Hugging Face).",
    "loaded": true,
    "artifact_path": "models/bert"
  },
  {
    "id": "both",
    "name": "Dual Comparison Mode",
    "description": "Run predictions on both TF-IDF and BERT models simultaneously for direct comparison.",
    "loaded": true,
    "artifact_path": "models/model.pkl"
  }
]
```

`both` is `loaded` only when **both** TF-IDF and BERT for that language are available.

### `POST /api/detect-lang`

Detect analysis language without running models.

**Request body:**

| Field | Type | Required | Description |
|-------|------|----------|-------------|
| `text` | string | yes | Comment text (1–8000 characters) |

```bash
curl -X POST http://127.0.0.1:8010/api/detect-lang \
  -H "Content-Type: application/json" \
  -d '{"text": "Dziękuję za pomoc!"}'
```

**Response:**

```json
{
  "analysis_lang": "pl",
  "confidence": 1.0,
  "is_reliable": true,
  "source": "gcld3",
  "detected_code": "pl"
}
```

`source` is one of `gcld3`, `langdetect`, `fallback`, or `forced` (when used via predict with explicit `lang`).

### `POST /api/predict`

Main inference endpoint.

**Request body:**

| Field | Type | Default | Description |
|-------|------|---------|-------------|
| `text` | string | — | Comment to classify (1–8000 chars) |
| `model` | string | `tfidf_lr` | `tfidf_lr`, `bert`, or `both` |
| `lang` | string | `auto` | `auto` (detect), `en`, or `pl` |

- **`lang: auto`** — routes to EN or PL models using gcld3 (Docker) or langdetect (local fallback).
- **`lang: en` / `pl`** — skips detection; `lang_source` in the response is `forced`.

```bash
curl -X POST http://127.0.0.1:8010/api/predict \
  -H "Content-Type: application/json" \
  -d '{
    "text": "I will find you and hurt you.",
    "model": "both",
    "lang": "auto"
  }'
```

**Response (single model, `model: bert`):**

```json
{
  "probabilities": {
    "toxic": 0.82,
    "severe_toxic": 0.15,
    "obscene": 0.04,
    "threat": 0.71,
    "insult": 0.09,
    "identity_hate": 0.02
  },
  "labels": ["toxic", "severe_toxic", "obscene", "threat", "insult", "identity_hate"],
  "model": "bert",
  "requested_lang": "auto",
  "analysis_lang": "en",
  "lang_confidence": 0.86,
  "lang_source": "gcld3",
  "is_dual": false,
  "similarity_projection": [
    {
      "id": "anchor_7",
      "text": "I will kill you...",
      "labels": ["toxic", "severe_toxic", "threat"],
      "x": 95.0,
      "y": 85.0,
      "similarity": 0.72,
      "is_active": false
    },
    {
      "id": "active_user",
      "text": "I will find you and hurt you.",
      "labels": ["toxic", "threat"],
      "x": 88.0,
      "y": 70.0,
      "similarity": 1.0,
      "is_active": true
    }
  ]
}
```

**Response extras when `model: both`:**

| Field | Description |
|-------|-------------|
| `is_dual` | `true` |
| `probabilities` | BERT/HerBERT scores (primary display) |
| `probabilities_tfidf` | TF-IDF scores |
| `probabilities_bert` | BERT/HerBERT scores |
| `similarity_projection_tfidf` | 2D map points for TF-IDF |
| `similarity_projection_bert` | 2D map points for BERT |

Polish (`analysis_lang: pl`) uses labels `safe`, `hate_speech`, `violence`, `vulgarity` instead of the six Jigsaw classes.

**HTTP errors:**

| Status | When |
|--------|------|
| **422** | Invalid body (empty text, unknown `model`) |
| **503** | Requested model not loaded (missing artifact) |
| **500** | Inference or internal error (detail includes message) |

### `GET /api/metrics`

Returns aggregated hold-out metrics for all four trained models:

```json
{
  "tfidf_lr": { "f1_macro": 0.64, "f1_micro": 0.77, "per_label": [...], "dataset": {...} },
  "bert": { "f1_macro": 0.69, "f1_micro": 0.81, "per_label": [...], "dataset": {...} },
  "tfidf_lr_pl": { ... },
  "bert_pl": { ... }
}
```

Data is read from `ml/experiments/*/metrics.json` (path configurable via `ML_EXPERIMENTS_DIR`). If files are missing, built-in fallback values are returned so the UI still renders.

Full backend notes: [docs/BACKEND.md](docs/BACKEND.md).

---

## Configuration

| Variable | Description |
|----------|-------------|
| `MODEL_PATH` | EN TF-IDF pickle |
| `BERT_MODEL_DIR` | EN BERT directory |
| `MODEL_PATH_PL` | PL TF-IDF pickle |
| `BERT_MODEL_DIR_PL` | PL HerBERT directory |
| `ML_EXPERIMENTS_DIR` | Path to `ml/experiments` |
| `LANG_DETECT_MIN_CONFIDENCE` | Auto language detection threshold |
| `VITE_PROXY_TARGET` | Backend URL for Vite (Docker) |

---

## Repository layout

```text
toxic-comment-detector/
├── backend/app/          # FastAPI (see docs/BACKEND.md)
├── frontend/src/         # React UI
├── ml/                   # Training (see docs/ML.md)
├── models/               # Artifacts + thresholds.json
├── data/raw/             # Jigsaw train.csv (not in git)
├── docs/                 # BACKEND.md, ML.md
├── images/demo/          # UI screenshots for README
└── docker-compose.yml
```

---

## License

This project is licensed under the **[MIT License](LICENSE)**.

```
Copyright (c) 2026 The Project Contributors
```

You may use, copy, modify, merge, publish, distribute, sublicense, and sell copies of the software, provided the copyright notice and permission notice are included in all copies or substantial portions.

The software is provided **"as is"**, without warranty of any kind.

### Third-party components

| Component | License / terms |
|-----------|-----------------|
| **Project source code** | MIT ([LICENSE](LICENSE)) |
| **scikit-learn, FastAPI, React, etc.** | See respective package licenses in `requirements.txt` / `package.json` |
| **BERT / HerBERT weights** | Subject to [Hugging Face model licenses](https://huggingface.co/models) (e.g. Apache 2.0 for `bert-base-uncased`, HerBERT terms for `allegro/herbert-base-cased`) |
| **Jigsaw dataset** | [Kaggle competition rules](https://www.kaggle.com/c/jigsaw-toxic-comment-classification-challenge) — not redistributed in this repo |
| **BAN-PL dataset** | Obtain separately; used for Polish model training only |

Training data and pretrained weights are **not** covered by the MIT license on this repository's code. Check each dataset and model card before commercial use or redistribution.
