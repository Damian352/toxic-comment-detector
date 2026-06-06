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

## API (summary)

| Method | Path | Description |
|--------|------|-------------|
| GET | `/api/health` | Liveness |
| GET | `/api/ready` | Model load status |
| GET | `/api/models?lang=en\|pl` | Model list |
| GET | `/api/metrics` | Metrics from `ml/experiments/` |
| POST | `/api/detect-lang` | Language detection |
| POST | `/api/predict` | Inference |

Example:

```json
POST /api/predict
{
  "text": "Your comment here",
  "model": "both",
  "lang": "auto"
}
```

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

See repository / contact the project owner.
