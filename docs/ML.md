# ML Pipeline Documentation

The `ml/` module handles **training**, **evaluation**, and **experiments**. Production inference lives in `backend/` but imports shared constants from `ml/labels.py` and utilities from `ml/training/bert_multilabel.py`.

## Structure

```text
ml/
├── labels.py                    # EN/PL taxonomies, thresholds from thresholds.json
├── preprocessing/text.py        # Text normalization for TF-IDF and BERT
├── evaluation/
│   ├── metrics.py               # multilabel_report (F1, Hamming, per-label)
│   └── threshold_tuning.py      # Per-label threshold search, report comparison
├── training/
│   ├── baseline_pipeline.py     # TF-IDF (word+char) + OvR LogisticRegression
│   ├── train_baseline.py        # Train EN TF-IDF → model.pkl
│   ├── train_baseline_pl.py     # Train PL TF-IDF → model_pl.pkl
│   ├── bert_multilabel.py       # Dataset, sigmoid, artifact save/load
│   ├── train_bert.py            # Fine-tune BERT → models/bert/
│   ├── train_bert_pl.py         # Fine-tune HerBERT → models/bert_pl/
│   └── tune_thresholds.py       # Tune thresholds for all 4 models
├── experiments/                 # metrics.json, thresholds.json per model
├── notebooks/01_eda.ipynb       # Jigsaw EDA
└── reports/                     # Stage reports
```

## Datasets

| Language | Source | Default path | Labels |
|----------|--------|--------------|--------|
| EN | [Jigsaw Toxic Comment](https://www.kaggle.com/c/jigsaw-toxic-comment-classification-challenge) | `data/raw/train.csv` | 6 toxic classes |
| PL | BAN-PL (Wykop.pl) | `BAN-PL_2/BAN-PL.csv` | safe + 3 violation types |

## Models

### 1. TF-IDF + Logistic Regression (baseline)

**Pipeline** (`baseline_pipeline.py`):

1. `FunctionTransformer` — preprocessing (`preprocess_batch` / `preprocess_batch_pl`).
2. `FeatureUnion` — word TF-IDF (1–2 grams) + char TF-IDF (3–5 grams, `char_wb`).
3. `OneVsRestClassifier(LogisticRegression)` — independent binary classifier per label.

**Why char n-grams:** catch obfuscation (`id!ot`, `f*ck`).

**Training:**

```bash
python -m ml.training.train_baseline              # EN, requires train.csv
python -m ml.training.train_baseline --demo       # demo corpus without Kaggle
python -m ml.training.train_baseline_pl           # PL, requires BAN-PL.csv
```

Artifacts: `models/model.pkl`, `models/model_pl.pkl`.

### 2. BERT / HerBERT (transformer)

**Architecture:** `AutoModelForSequenceClassification` with `problem_type="multi_label_classification"`. At inference — **sigmoid** per logit (not softmax).

| Language | Base model | Script | Output |
|----------|------------|--------|--------|
| EN | `bert-base-uncased` | `train_bert.py` | `models/bert/` |
| PL | `allegro/herbert-base-cased` | `train_bert_pl.py` | `models/bert_pl/` |

```bash
python -m ml.training.train_bert --demo --epochs 3
python -m ml.training.train_bert_pl --epochs 3
```

Each directory stores weights, tokenizer, and `labels.json` (label order, `max_length`).

## Preprocessing

**EN** (`preprocess_text`): HTML unescape → lowercase → strip URLs/HTML → collapse whitespace.

**PL** (`preprocess_text_pl`): remove `{USERNAME}`, `{URL}`, `&gt;`, URLs, normalize whitespace. No aggressive lemmatization — preserve vulgarity signals.

## Evaluation

`multilabel_report(y_true, y_pred, label_names)` returns:

- `hamming_loss`, `f1_macro`, `f1_micro`, precision/recall macro/micro
- `per_label[]` — precision, recall, f1, support
- `classification_report` — sklearn text report

Binary predictions from probabilities: default threshold 0.5 or per-label from `threshold_tuning.py`.

## Threshold tuning

`python -m ml.training.tune_thresholds`

1. Same outer test split as training (`test_size=0.2`, `random_state=42`).
2. Inner validation (20% of train) — grid search thresholds 0.05–0.95 per label (max F1).
3. Re-evaluate on hold-out test with tuned thresholds.
4. Write to `models/thresholds.json` and update `ml/experiments/*/metrics.json`.

**No retraining required** — only decision thresholds change.

## Shared constants (`labels.py`)

```python
LABELS      # EN: 6 toxic classes
PL_LABELS   # PL: safe, hate_speech, violence, vulgarity
DEFAULT_THRESHOLD = 0.5
```

`get_per_label_thresholds(lang, model_id, labels)` — reads `models/thresholds.json`.

`active_labels_from_probs(probs, thresholds, lang)` — human-readable active labels for UI/API:

- **PL:** violation labels take priority; if none active → `safe`.
- **EN:** any label above threshold; if none → `safe`.

## Dependencies

```bash
pip install -r ml/requirements.txt
```

BERT requires PyTorch + transformers. Baseline requires scikit-learn, pandas, numpy.

## Typical workflow

```bash
# 1. Data
#    data/raw/train.csv (Kaggle)
#    BAN-PL_2/BAN-PL.csv (Polish)

# 2. Training
python -m ml.training.train_baseline
python -m ml.training.train_bert --epochs 2
python -m ml.training.train_baseline_pl
python -m ml.training.train_bert_pl --epochs 2

# 3. Thresholds
python -m ml.training.tune_thresholds

# 4. Backend
cd backend && uvicorn app.main:app --port 8010
```

## Experiments

Results in `ml/experiments/<experiment_name>/`:

- `metrics.json` — metrics with tuned thresholds + `baseline_at_0.5` for comparison
- `thresholds.json` — copy of per-label thresholds (duplicates registry)

Reports: `ml/reports/01_tfidf_logistic_regression_baseline.md`, `02_nlp_models_implementation.md`.
