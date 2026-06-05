# Toxic Comment Detector

A multi-component workspace for **multi-label toxic comment classification** in natural language. The goal is to detect offensive, aggressive, or hateful content where a single comment may belong to **several toxicity categories at once** (e.g. `toxic` and `insult`).

This repository separates **research/training** (`ml/`), **production-style inference API** (`backend/`), and a **small React UI** (`frontend/`), with optional **Docker Compose** to run the API and UI together.

---

## Current status

| Area | Status |
|------|--------|
| **ML** | Two approaches: **TF-IDF + Logistic Regression** (`train_baseline.py` → `models/model.pkl`) and **BERT** (`train_bert.py` → `models/bert/`). NLP report: `ml/reports/02_nlp_models_implementation.md`. |
| **Backend** | FastAPI loads both artifacts (when present), `/api/models`, `/api/predict` with `model`: `tfidf_lr` or `bert`. |
| **Frontend** | React UI with model selector, `/api/predict`, per-label probability bars. |
| **Docker** | `docker-compose.yml` runs backend (port 8000) and frontend dev server (5173) with hot reload and API proxying. |

The default label order matches the **Jigsaw Toxic Comment Classification** challenge:

`toxic`, `severe_toxic`, `obscene`, `threat`, `insult`, `identity_hate`

---

## Repository layout

```text
toxic-comment-detector/
├── backend/                 # FastAPI: model loading + inference only
│   ├── app/
│   │   ├── main.py          # App factory, CORS, lifespan (load model)
│   │   ├── api/routes.py    # /api/health, /api/ready, /api/predict
│   │   ├── core/config.py   # MODEL_PATH, CORS origins
│   │   └── services/inference.py
│   ├── Dockerfile
│   └── requirements.txt
├── frontend/                # React + Vite UI
│   ├── Dockerfile
│   ├── vite.config.ts       # Dev proxy: /api → backend
│   └── src/
├── ml/                      # Training, experiments, notebooks, metrics
│   ├── training/
│   │   └── train_baseline.py
│   ├── evaluation/
│   │   └── metrics.py       # Multilabel F1 / precision / recall helpers
│   ├── experiments/
│   ├── notebooks/
│   │   └── 01_eda.ipynb     # Stage-1 EDA (Kaggle train.csv)
│   └── requirements.txt
├── data/
│   ├── raw/                 # Place Kaggle train.csv here (not committed)
│   └── processed/
├── models/                  # Serialized model for backend (e.g. model.pkl)
├── docker-compose.yml
├── .gitignore
└── README.md
```

**Why this split**

- **`ml/`** — everything that touches dataset exploration, training, comparing approaches, and offline evaluation. Keeps the API image small and avoids leaking training dependencies into production.
- **`backend/`** — only what is needed at **inference time**: load artifact, run `predict_proba`, expose HTTP.
- **`frontend/`** — quick manual testing and demo of probabilities per class.

---

## Prerequisites

- **Python** 3.12+ recommended (3.11+ should work for the pinned stack).
- **Node.js** 18+ (Vite 5; avoid very old Node for the toolchain).
- **Docker Desktop** (optional) if you use Compose.

---

## Quick start (local, no Docker)

### 1. Model artifact

From the **repository root**:

```bash
python -m pip install -r ml/requirements.txt
python -m ml.training.train_baseline
python -m ml.training.train_bert --demo --epochs 3
```

This writes **`models/model.pkl`** (TF-IDF + LR) and **`models/bert/`** (transformer). Demo scripts work without Kaggle data; for Jigsaw metrics use `data/raw/train.csv` and drop `--demo`.

### 2. Backend

```bash
cd backend
python -m pip install -r requirements.txt
python -m uvicorn app.main:app --reload --host 127.0.0.1 --port 8000
```

- **Default model path**: `../models/model.pkl` relative to the repo when `MODEL_PATH` is unset (see `backend/app/core/config.py`).
- **Override**: set environment variable `MODEL_PATH` to an absolute or relative path to your pickle.

Useful endpoints:

| Method | Path | Purpose |
|--------|------|---------|
| GET | `/` | Short service info |
| GET | `/docs` | Swagger UI |
| GET | `/api/health` | Liveness |
| GET | `/api/models` | Available models and load status |
| GET | `/api/ready` | Load status for TF-IDF and BERT artifacts |
| POST | `/api/predict` | JSON `{"text":"...","model":"tfidf_lr"}` or `"bert"` → probabilities |

### 3. Frontend

In another terminal, from **`frontend/`**:

```bash
npm install
npm run dev
```

Open `http://127.0.0.1:5173`. The Vite dev server **proxies** `http://127.0.0.1:8000` for paths starting with `/api`, so the browser can call `/api/predict` without CORS issues.

OpenAPI from the UI link uses `VITE_BACKEND_ORIGIN` (default `http://127.0.0.1:8000`).

---

## Docker Compose

From the **repository root**:

```bash
docker compose up --build
```

- **Backend**: `http://localhost:8000` — mounts `./models` as `/models` and sets `MODEL_PATH=/models/model.pkl`.
- **Frontend**: `http://localhost:5173` — bind-mounts `./frontend`, runs `npm install` then `npm run dev`.  
  - `VITE_PROXY_TARGET=http://backend:8000` so `/api` from the browser hits the Vite server, which forwards to the backend container.
  - `VITE_BACKEND_ORIGIN=http://localhost:8000` so the “OpenAPI docs” link still works from your host browser.

Ensure **`models/model.pkl`** exists on the host before expecting successful predictions (train locally or copy an artifact in).

---

## Data and EDA (Stage 1)

1. Download **`train.csv`** from the [Kaggle Jigsaw Toxic Comment Classification Challenge](https://www.kaggle.com/c/jigsaw-toxic-comment-classification-challenge).
2. Save it as **`data/raw/train.csv`** (repo root).

Then:

```bash
python -m pip install -r ml/requirements.txt
jupyter lab ml/notebooks/01_eda.ipynb
```

The notebook discovers the repo root, loads the CSV, and produces class balance, length distributions, multi-label statistics, toxic vs non-toxic share, and sample comments for reporting.

---

## Configuration reference

| Variable | Where | Purpose |
|----------|--------|---------|
| `MODEL_PATH` | Backend | Path to pickled sklearn-compatible pipeline |
| `VITE_PROXY_TARGET` | Frontend (Docker) | Backend URL for Vite proxy |
| `VITE_BACKEND_ORIGIN` | Frontend | Base URL shown for OpenAPI link |

Backend CORS defaults include `http://localhost:5173` and `http://127.0.0.1:5173`; extend in `backend/app/core/config.py` if needed.

---

## Development notes

- **Git** ignores large artifacts (`models/*.pkl`), virtualenvs, `node_modules`, Jupyter checkpoints, and typical IDE/OS noise — see `.gitignore`.
- **Baseline model** is for wiring only; metrics on toy data are not meaningful. Train on Jigsaw (or similar) in `ml/training/` for real evaluation using `ml/evaluation/metrics.py`.
- **API contract**: `POST /api/predict` body `{"text": "..."}` returns `probabilities` (map label → float) and `labels` (ordered list of class names).

---

## License
