# NLP in Toxic Comment Detector: TF-IDF + LR and BERT

**Date:** 2026-06-04  
**Dataset:** [Jigsaw Toxic Comment Classification Challenge](https://www.kaggle.com/c/jigsaw-toxic-comment-classification-challenge) (`data/raw/train.csv`)  
**Artifacts:** `models/model.pkl` (baseline), `models/bert/` (transformer)  
**Metrics:** `ml/experiments/baseline_tfidf_lr/metrics.json`, `ml/experiments/bert_multilabel/metrics.json`

---

## 1. Problem Statement (NLP)

**Multi-label text classification** of comments: for each of six binary attributes, the model outputs a probability \(P(y_i = 1 \mid \text{text})\). A single comment can have multiple labels at once (for example, `toxic` + `insult` + `obscene`).

Fixed label order (Jigsaw):

| # | Label | Meaning (brief) |
|---|--------|----------------|
| 1 | `toxic` | General toxicity |
| 2 | `severe_toxic` | Severe toxicity |
| 3 | `obscene` | Profane language |
| 4 | `threat` | Threats |
| 5 | `insult` | Insults |
| 6 | `identity_hate` | Hostility toward a social group |

Source of truth for label order: `ml/labels.py` (used in training, evaluation, and the API).

---

## 2. Overview of Both Approaches

```mermaid
flowchart TB
    subgraph input [Input]
        T[comment_text]
    end

    subgraph classic [Approach 1: TF-IDF + LR]
        T --> P1[preprocess_text]
        P1 --> W[Word TF-IDF 1-2]
        P1 --> C[Char TF-IDF 3-5]
        W --> F[Sparse matrix]
        C --> F
        F --> OVR[6 × Logistic Regression OvR]
        OVR --> S1[6 probabilities]
    end

    subgraph neural [Approach 2: BERT]
        T --> TOK[BERT WordPiece tokenizer]
        TOK --> ENC[768-d contextual embeddings]
        ENC --> HEAD[Linear head + sigmoid]
        HEAD --> S2[6 probabilities]
    end

    S1 --> API[FastAPI /api/predict]
    S2 --> API
    API --> UI[React: model selection]
```

| Aspect | TF-IDF + LR | BERT |
|--------|-------------|------|
| Text representation | Sparse frequency n-grams | Contextual subword embeddings (WordPiece) |
| Model | Linear (OvR) | Transformer encoder + linear head |
| Training | `sklearn` Pipeline, CPU | `transformers` Trainer, GPU optional |
| Artifact | `model.pkl` | `models/bert/` (HF format) |
| API/UI ID | `tfidf_lr` | `bert` |

---

## 3. Approach 1: TF-IDF + Logistic Regression

Detailed report: `ml/reports/01_tfidf_logistic_regression_baseline.md`.

### 3.1. Preprocessing (`ml/preprocessing/text.py`)

Before vectorization, **light normalization** is applied (no lemmatization or stemming):

1. HTML unescape (`&amp;` → `&`)
2. Lowercasing
3. URL and HTML tag removal
4. Whitespace collapsing

This reduces noise and feature duplication while preserving obfuscated spellings for char n-grams.

### 3.2. Features

**Word TF-IDF:** `analyzer="word"`, n-grams (1, 2), `max_features=100_000`, `sublinear_tf=True` — captures lexicon and short phrases («shut up»).

**Char TF-IDF:** `analyzer="char_wb"`, n-grams (3, 5) — robust to `1diot`, `id!ot`.

Both streams are combined via `FeatureUnion`.

### 3.3. Classifier

`OneVsRestClassifier(LogisticRegression)` — six independent binary LRs with `class_weight="balanced"`. At inference: `predict_proba` → \(P(y=1)\) for each label.

### 3.4. Training

```bash
python -m ml.training.train_baseline
```

Export: `models/model.pkl`. Metrics: Hamming loss, macro/micro F1, per-label P/R/F1 (`ml/evaluation/metrics.py`).

---

## 4. Approach 2: BERT (Multi-Label)

### 4.1. Why BERT for Toxic Detection

- **Context:** the word «kill» in «kill the lights» vs a threat — contextual embeddings account for surrounding text.
- **Synonyms and paraphrasing:** no need for an exact matching n-gram in the training set.
- **Transfer learning:** `bert-base-uncased` is pretrained on a large English corpus.

Trade-offs: more memory, slower inference, requires `torch` and `transformers`.

### 4.2. Architecture

- Base model: **`bert-base-uncased`** (12 layers, 768 hidden, WordPiece, max 512 tokens; **256** in this project for inference/training speed).
- Head: `AutoModelForSequenceClassification` with `problem_type="multi_label_classification"`, `num_labels=6`.
- Loss function (inside HF): **BCEWithLogitsLoss** across all labels.
- Output probabilities: **sigmoid** per logit (independent labels).

Model assembly code: `ml/training/bert_multilabel.py` → `build_bert_model()`.

### 4.3. Preprocessing for BERT

Text is fed to **`AutoTokenizer`** with almost **no** sklearn preprocessing: the tokenizer splits into subwords, adds `[CLS]` / `[SEP]`, and applies truncation/padding up to `max_length`.

For the TF-IDF baseline, preprocessing remains mandatory; for BERT this is an intentional pipeline difference (typical practice for transformers).

### 4.4. Training

```bash
# Quick check on demo corpus (~10 examples)
python -m ml.training.train_bert --demo --epochs 3

# Full Jigsaw (slow; subsample on CPU)
python -m ml.training.train_bert --max-samples 5000 --epochs 1

# Full training (GPU recommended)
python -m ml.training.train_bert --epochs 2 --batch-size 16
```

Script: `ml/training/train_bert.py`

- Data loading: same scheme as baseline (`load_jigsaw_dataset` from `train_baseline.py`).
- Split: `train_test_split`, stratify on `toxic` column.
- `Trainer` (Hugging Face): AdamW, `learning_rate=2e-5`, `weight_decay=0.01`, eval every epoch.
- Hold-out metrics: same `multilabel_report`, default threshold **0.5**.
- Export: `models/bert/` (`config.json`, weights, tokenizer, `labels.json` with label order).

### 4.5. Inference (Backend)

`backend/app/services/bert_inference.py`:

1. Load `AutoModelForSequenceClassification` and tokenizer from `BERT_MODEL_DIR`.
2. Tokenize a single comment, `max_length` from `labels.json` (or 256).
3. Forward pass, sigmoid over logits → dictionary `{label: probability}`.

---

## 5. API and UI

### 5.1. Model Selection

| Endpoint | Purpose |
|----------|------------|
| `GET /api/models` | List models, description, `loaded` flag |
| `POST /api/predict` | Body: `{ "text": "...", "model": "tfidf_lr" \| "bert" }` |
| `GET /api/ready` | Load status for both models |

Registry: `backend/app/services/registry.py` — on startup, attempts to load both artifacts; a missing file does not crash the app but is marked `loaded: false`.

Environment variables:

- `MODEL_PATH` — path to `model.pkl`
- `BERT_MODEL_DIR` — `models/bert` directory

### 5.2. Frontend

`frontend/src/App.tsx`: radio buttons for `tfidf_lr` / `bert`, `GET /api/models` on mount, `model` field passed in `POST /api/predict`. Unavailable models are shown as «not loaded».

---

## 6. Approach Comparison from an NLP Perspective

| Criterion | TF-IDF + LR | BERT |
|----------|-------------|------|
| Inductive bias | Linear combinations of n-grams | Nonlinear contextual representations |
| OOV / rare words | Depends on n-grams in train | Subword (WordPiece) |
| Obfuscation (`id!ot`) | Char n-grams | Subwords + context |
| Irony / negation | Weak | Stronger (not guaranteed) |
| Training speed | Minutes on CPU | Hours without GPU / minutes with GPU |
| Inference speed | Milliseconds | Tens–hundreds of ms on CPU |
| Interpretability | Weights per n-gram | Low (attention not exposed in API) |

On Jigsaw hold-out, the TF-IDF+LR baseline gives a reference **F1 macro ≈ 0.62** (see `baseline_tfidf_lr/metrics.json`). BERT after full fine-tuning typically yields a **macro-F1 gain**, especially on rare classes (`threat`, `identity_hate`), but exact numbers depend on epochs, `max_samples`, and hardware — check `bert_multilabel/metrics.json` after your run.

---

## 7. NLP Code Structure in the Repository

```text
ml/
├── labels.py                      # 6 Jigsaw labels
├── preprocessing/text.py          # Preprocessing for TF-IDF
├── training/
│   ├── baseline_pipeline.py       # sklearn Pipeline
│   ├── train_baseline.py          # TF-IDF+LR training
│   ├── bert_multilabel.py         # Dataset, sigmoid, save artifact
│   └── train_bert.py              # Fine-tune BERT
├── evaluation/metrics.py          # Multilabel metrics
└── reports/
    ├── 01_tfidf_logistic_regression_baseline.md
    └── 02_nlp_models_implementation.md   # this document

backend/app/services/
├── inference.py                   # sklearn predict_proba
├── bert_inference.py              # HF BERT predict
└── registry.py                    # routing by model id
```

---

## 8. Recommended Startup Order

```bash
# 1. ML dependencies (from repository root)
pip install -r ml/requirements.txt

# 2. Baseline (if model.pkl does not exist yet)
python -m ml.training.train_baseline

# 3. BERT (demo first, then full dataset)
python -m ml.training.train_bert --demo --epochs 3
# python -m ml.training.train_bert   # full Jigsaw

# 4. Backend + frontend
pip install -r backend/requirements.txt
cd backend && python -m uvicorn app.main:app --reload --host 127.0.0.1 --port 8010
cd frontend && npm run dev
```

In the UI, select a model and submit a comment — the response includes `probabilities` and a `model` field confirming which backend was used.

---

## 9. Possible NLP Improvements

- **Unified preprocessing experiment** for BERT (light normalization before the tokenizer).
- Models **`unitary/toxic-bert`** or **`roberta-base`** as an alternative `--pretrained`.
- **Focal loss** / rare-class weighting for BERT.
- Probability calibration (temperature scaling) for comparable 0.5 thresholds across models.
- Export BERT to **ONNX** for faster CPU inference.

---

*This document describes the implementation of two NLP pipelines in the toxic-comment-detector project and their integration into the API/UI.*
