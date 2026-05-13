# Toxic Comment Detector

A multi-component workspace for **multi-label toxic comment classification** in natural language. The goal is to detect offensive, aggressive, or hateful content where a single comment may belong to **several toxicity categories at once** (e.g. `toxic` and `insult`).

This repository separates **research/training** (`ml/`), **production-style inference API** (`backend/`), and a **small React UI** (`frontend/`), with optional **Docker Compose** to run the API and UI together.

---

## Current status

| Area | Status |
|------|--------|
| **ML** | Baseline training script (`ml/training/train_baseline.py`) exports a scikit-learn pipeline to `models/model.pkl`. EDA notebook `ml/notebooks/01_eda.ipynb` expects Kaggle Jigsaw `train.csv` under `data/raw/`. Evaluation helpers live under `ml/evaluation/`. |
| **Backend** | FastAPI app loads `model.pkl`, serves health/ready and `/api/predict` with per-label probabilities. CORS enabled for local Vite. |
| **Frontend** | React + Vite + TypeScript: text input, call to `/api/predict`, bar display of scores; link to OpenAPI docs. |
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
python ml/training/train_baseline.py
```

This writes **`models/model.pkl`** (gitignored except structure). The script uses a tiny synthetic demo corpus so the stack is runnable without external data; replace with a real trained model when ready.

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
| GET | `/api/ready` | Whether `model.pkl` was found and loaded |
| POST | `/api/predict` | JSON `{"text":"..."}` → per-label probabilities |

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
