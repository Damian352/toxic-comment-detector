# Approach 1: TF-IDF + Logistic Regression — Baseline for Toxic Comments

**Date:** 2026-05-27  
**Dataset:** [Jigsaw Toxic Comment Classification Challenge](https://www.kaggle.com/c/jigsaw-toxic-comment-classification-challenge) (`data/raw/train.csv`)  
**Model artifact:** `models/model.pkl`  
**Experiment metrics:** `ml/experiments/baseline_tfidf_lr/metrics.json`

---

## 1. Goal and Context

The task is **multi-label classification** of comments: a single text can belong to several toxicity categories at once (for example, `toxic` + `insult` + `obscene`).

The first approach is a classic baseline:

> **Preprocessing → Word TF-IDF + Character TF-IDF → OneVsRest(Logistic Regression)**

This is a strong starting point for toxic detection because:

- short texts are well described by frequency-based n-gram features;
- toxicity is often expressed through **specific words and phrases** («idiot», «shut up», «go to hell»);
- character n-grams catch **intentionally distorted** words (`1diot`, `id!ot`, `i.d.i.o.t`);
- Logistic Regression works well on **sparse** TF-IDF vectors, trains quickly, and provides **probabilities** for the UI/API.

---

## 2. Why TF-IDF + Logistic Regression

### 2.1. TF-IDF on Toxic Texts

TF-IDF (Term Frequency — Inverse Document Frequency) increases the weight of rare but class-characteristic tokens:

| Example | What TF-IDF sees |
|--------|------------------|
| «you idiot» | high weight for unigram `idiot` |
| «shut up» | high weight for bigram `shut up` |
| «id!ot» (char 3–5) | patterns `idi`, `dio`, `iot` |

**Word n-grams (1–2):** capture profanity, offensive vocabulary, and typical toxic patterns.  
**Character n-grams (3–5, `char_wb`):** robust to obfuscation and punctuation within a word.

### 2.2. Why Logistic Regression, Not Random Forest

| Model | Problem for text |
|--------|---------------------|
| **Random Forest** | poor on sparse TF-IDF, huge dimensionality, slow, weak signal |
| **Linear SVM** | strong on sparse vectors, but probabilities are less convenient |
| **Logistic Regression** | linear model on sparse features, fast, interpretable, `predict_proba` out of the box |

**Conclusion:** for a baseline, **TF-IDF + Logistic Regression** is optimal.

### 2.3. Why Multi-Label, Not Multiclass

A comment like *«You stupid disgusting idiot»* can simultaneously be:

- `toxic`
- `insult`
- `obscene`

Therefore **multi-label binary classification** is used: a separate binary classifier per label via `OneVsRestClassifier`.

---

## 3. Pipeline Architecture

```mermaid
flowchart LR
    A[comment_text] --> B[preprocess_text]
    B --> C[FeatureUnion]
    C --> D[word_tfidf<br/>unigrams + bigrams]
    C --> E[char_tfidf<br/>char_wb 3-5]
    D --> F[Sparse feature matrix]
    E --> F
    F --> G[OneVsRestClassifier]
    G --> H1[LR: toxic]
    G --> H2[LR: severe_toxic]
    G --> H3[LR: obscene]
    G --> H4[LR: threat]
    G --> H5[LR: insult]
    G --> H6[LR: identity_hate]
    H1 --> I[P(y=1) per label]
    H2 --> I
    H3 --> I
    H4 --> I
    H5 --> I
    H6 --> I
```

### 3.1. sklearn Pipeline Schema

```
Pipeline([
  ("preprocess", FunctionTransformer(preprocess_batch)),
  ("features", FeatureUnion([
      ("word_tfidf", TfidfVectorizer(analyzer="word", ngram_range=(1, 2), ...)),
      ("char_tfidf", TfidfVectorizer(analyzer="char_wb", ngram_range=(3, 5), ...)),
  ])),
  ("clf", OneVsRestClassifier(LogisticRegression(...))),
])
```

At inference, the backend calls only:

```python
model.predict_proba([text])  # → probabilities for 6 labels
```

---

## 4. Project Files and Functions

### 4.1. ML Code Structure

| File | Purpose |
|------|------------|
| `ml/preprocessing/text.py` | Text preprocessing functions |
| `ml/training/baseline_pipeline.py` | sklearn Pipeline assembly |
| `ml/training/train_baseline.py` | CLI: data loading, training, metrics, export |
| `ml/evaluation/metrics.py` | Multi-label metrics (F1, Hamming loss, per-label report) |
| `ml/experiments/baseline_tfidf_lr/metrics.json` | JSON with hold-out evaluation results |
| `models/model.pkl` | Serialized Pipeline for FastAPI |
| `backend/app/services/inference.py` | Pickle loading and `predict_proba` in the API |

### 4.2. `ml/preprocessing/text.py`

**`preprocess_text(text: str) -> str`**

Applied steps:

1. HTML unescape (`&amp;` → `&`)
2. **lowercase**
3. **URL** removal (`http://...`, `www....`)
4. **HTML tag** removal
5. **whitespace** normalization

**What we intentionally do NOT do:**

| Action | Reason |
|----------|---------|
| Full punctuation removal | `!!!` can signal aggression |
| Stopword removal | «go to **hell**» — stopwords matter for toxicity |
| Aggressive stemming | toxic patterns can be lost |

**`preprocess_batch(texts)`** — wrapper for sklearn `FunctionTransformer`, accepts an iterable of strings.

### 4.3. `ml/training/baseline_pipeline.py`

**`build_baseline_pipeline(...) -> Pipeline`**

Default hyperparameters:

| Component | Parameter | Value | Meaning |
|-----------|----------|----------|-------|
| Word TF-IDF | `ngram_range` | `(1, 2)` | unigrams + bigrams |
| Word TF-IDF | `max_features` | `100_000` | vocabulary size limit |
| Char TF-IDF | `analyzer` | `char_wb` | char n-grams with word boundaries |
| Char TF-IDF | `ngram_range` | `(3, 5)` | robustness to obfuscation |
| Both TF-IDF | `min_df` | `5` | filter rare noise |
| Both TF-IDF | `max_df` | `0.95` | filter overly frequent tokens |
| Both TF-IDF | `sublinear_tf` | `True` | `1 + log(tf)` — softens frequent repeats |
| Logistic Regression | `solver` | `liblinear` | fast and stable on sparse TF-IDF |
| Logistic Regression | `class_weight` | `balanced` | compensates for class imbalance |
| Logistic Regression | `C` | `1.0` | standard L2 regularization |
| Logistic Regression | `max_iter` | `2000` | optimization iteration limit |

**Labels (fixed order, matches backend):**

`toxic`, `severe_toxic`, `obscene`, `threat`, `insult`, `identity_hate`

### 4.4. `ml/training/train_baseline.py`

**`load_jigsaw_dataset(path)`** — reads CSV, returns `(texts, y)` where `y` has shape `(n_samples, 6)`.

**`train_and_evaluate(...)`** — train/test split, fit, hold-out metrics.

**CLI:**

```bash
# Full training on Jigsaw train.csv
python -m ml.training.train_baseline --data data/raw/train.csv

# Quick smoke test on demo corpus (no Kaggle data)
python -m ml.training.train_baseline --demo
```

Arguments:

| Flag | Default | Description |
|------|--------------|----------|
| `--data` | `data/raw/train.csv` | Path to CSV |
| `--out` | `models/model.pkl` | Where to save the model |
| `--metrics-out` | `ml/experiments/baseline_tfidf_lr/metrics.json` | JSON with metrics |
| `--test-size` | `0.2` | Hold-out fraction |
| `--threshold` | `0.5` | Threshold for binary predictions during evaluation |
| `--demo` | — | Force demo corpus |

### 4.5. `ml/evaluation/metrics.py`

**`multilabel_report(y_true, y_pred, label_names)`** returns:

- `hamming_loss`
- `f1_macro`, `f1_micro`
- `precision_macro`, `precision_micro`
- `recall_macro`, `recall_micro`
- `per_label` — precision/recall/F1/support per label
- `classification_report` — sklearn text report

### 4.6. Backend Integration

`backend/app/services/inference.py`:

- **`ToxicInferenceService`** loads `model.pkl` at FastAPI startup;
- **`predict_proba(text)`** returns `dict[label, float]`;
- **`_scores_from_predict_proba`** normalizes different sklearn output formats (list vs ndarray).

API endpoint: `POST /api/predict` with body `{"text": "..."}`.

---

## 5. Training Process (Actual Run)

### 5.1. Data

| Parameter | Value |
|----------|----------|
| Source | `data/raw/train.csv` |
| Number of comments | **159 571** |
| Train split | **127 656** (80%) |
| Test split | **31 915** (20%) |
| Stratify | on `toxic` column |
| Share of comments with ≥1 label | **~10.2%** |

### 5.2. Training Time

On a local machine (Windows, scikit-learn + liblinear): **~3.3 minutes** on the full dataset.

> Note: the `saga` solver was too slow at this scale; `liblinear` is the standard, practical choice for sparse TF-IDF + OvR.

### 5.3. Threshold and Metrics

For offline evaluation, binary predictions use a **0.5** threshold on `predict_proba`.  
For production/UI, the threshold can be calibrated separately per label (especially for rare classes).

---

## 6. Hold-Out Evaluation Results (test 20%)

### 6.1. Overall Metrics

| Metric | Value |
|---------|----------|
| **Hamming loss** | **0.0232** |
| **F1 macro** | **0.6162** |
| **F1 micro** | **0.7330** |
| Precision macro | 0.5160 |
| Precision micro | 0.6369 |
| Recall macro | 0.8025 |
| Recall micro | 0.8632 |

**Interpretation:**

- **F1 micro ≈ 0.73** — a solid baseline on Jigsaw; the model confidently catches obvious toxicity.
- **Recall > Precision** (especially on rare classes) — the model tends to **over-predict** (many false positives), which is typical with `class_weight="balanced"` and a 0.5 threshold.
- **Hamming loss ≈ 0.023** — on average ~2.3% of labels are wrong per comment.

### 6.2. Per-Label F1

| Label | Precision | Recall | F1 | Support (test) |
|-------|-----------|--------|-----|----------------|
| **toxic** | 0.717 | 0.873 | **0.787** | 3059 |
| **obscene** | 0.753 | 0.890 | **0.816** | 1710 |
| **insult** | 0.618 | 0.860 | **0.719** | 1590 |
| **identity_hate** | 0.338 | 0.768 | **0.469** | 289 |
| **severe_toxic** | 0.319 | 0.794 | **0.455** | 311 |
| **threat** | 0.351 | 0.629 | **0.450** | 97 |

**Observations:**

- Best performance on **`obscene`** and **`toxic`** — many lexical markers.
- **`insult`** — solid mid-tier result (F1 ≈ 0.72).
- **`severe_toxic`**, **`threat`**, **`identity_hate`** — rare classes with low precision; the model finds them (high recall) but often fires false positives.

### 6.3. Qualitative Examples (Inference)

After training, the model correctly returns probabilities through the backend service:

- *«You are an idiot, shut up!»* → high scores for `toxic`, `insult`, `obscene`
- *«Thank you for the helpful article.»* → low scores across all labels

---

## 7. How It Works End-to-End

```
1. User enters text in the React UI (frontend/)
2. POST /api/predict → FastAPI (backend/)
3. ToxicInferenceService.predict_proba(text)
4. Pipeline:
     preprocess_batch → FeatureUnion(word+char TF-IDF) → 6× LogisticRegression
5. JSON response: { "probabilities": { "toxic": 0.92, ... }, "labels": [...] }
6. UI renders a bar chart per label
```

The model is serialized via **pickle** — the same Pipeline object trained in `ml/training/`, with no separate conversion step.

---

## 8. Baseline Limitations and Next Steps

### 8.1. TF-IDF Limitations

TF-IDF **does not understand semantics**:

| Text A | Text B | TF-IDF |
|---------|---------|--------|
| «You are dumb» | «You are stupid» | different vectors, same meaning |

It does not catch:

- sarcasm without toxic lexicon;
- veiled aggression («I hope you have a *very* special day»);
- cross-lingual / code-switching without char patterns.

### 8.2. Possible Improvements (Approach 2+)

1. **Per-label threshold calibration** on a validation set (improves precision on rare classes).
2. **Linear SVM** as an alternative linear baseline for comparison.
3. **Embeddings / Transformers** (BERT, DistilBERT) — semantic understanding.
4. **Ensemble** of TF-IDF + transformer scores.
5. **Cross-validation** instead of a single hold-out split for more stable metrics.

---

## 9. Quick Start (Reproduce the Experiment)

```bash
# from repository root
python -m pip install -r ml/requirements.txt

# place train.csv in data/raw/ (Kaggle Jigsaw)

python -m ml.training.train_baseline --data data/raw/train.csv

# start the API
cd backend
python -m pip install -r requirements.txt
python -m uvicorn app.main:app --reload --host 127.0.0.1 --port 8000
```

---

## 10. Summary

| Component | Solution |
|-----------|---------|
| Features | Word TF-IDF (1–2) + Char TF-IDF (3–5, `char_wb`) |
| Classifier | `OneVsRestClassifier(LogisticRegression)` |
| Multi-label | 6 independent binary LRs |
| Preprocessing | lowercase, URL/HTML removal, whitespace; punctuation preserved |
| Dataset | Jigsaw, 159 571 comments |
| Best F1 | `obscene` (0.82), `toxic` (0.79) |
| Overall F1 micro | **0.73** |
| Artifact | `models/model.pkl` |

The baseline is implemented, trained on real data, integrated with the FastAPI backend, and ready as a **strong starting point** before embedding/transformer approaches.
