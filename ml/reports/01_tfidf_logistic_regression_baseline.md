# Подход 1: TF-IDF + Logistic Regression — baseline для токсичных комментариев

**Дата:** 2026-05-27  
**Датасет:** [Jigsaw Toxic Comment Classification Challenge](https://www.kaggle.com/c/jigsaw-toxic-comment-classification-challenge) (`data/raw/train.csv`)  
**Артефакт модели:** `models/model.pkl`  
**Метрики эксперимента:** `ml/experiments/baseline_tfidf_lr/metrics.json`

---

## 1. Цель и контекст

Задача — **multi-label классификация** комментариев: один текст может одновременно относиться к нескольким категориям токсичности (например, `toxic` + `insult` + `obscene`).

Первый подход — классический baseline:

> **Препроцессинг → Word TF-IDF + Character TF-IDF → OneVsRest(Logistic Regression)**

Это сильная отправная точка для toxic detection, потому что:

- короткие тексты хорошо описываются частотными n-gram признаками;
- токсичность часто выражается **конкретными словами и фразами** («idiot», «shut up», «go to hell»);
- character n-grams ловят **намеренно искажённые** слова (`1diot`, `id!ot`, `i.d.i.o.t`);
- Logistic Regression хорошо работает на **разреженных** TF-IDF-векторах, быстро обучается и даёт **вероятности** для UI/API.

---

## 2. Почему именно TF-IDF + Logistic Regression

### 2.1. TF-IDF на toxic-текстах

TF-IDF (Term Frequency — Inverse Document Frequency) повышает вес редких, но характерных для класса токенов:

| Пример | Что видит TF-IDF |
|--------|------------------|
| «you idiot» | высокий вес unigram `idiot` |
| «shut up» | высокий вес bigram `shut up` |
| «id!ot» (char 3–5) | паттерны `idi`, `dio`, `iot` |

**Word n-grams (1–2):** ловят ругательства, offensive vocabulary, типичные toxic-паттерны.  
**Character n-grams (3–5, `char_wb`):** устойчивы к обфускации и пунктуации внутри слова.

### 2.2. Почему Logistic Regression, а не Random Forest

| Модель | Проблема для текста |
|--------|---------------------|
| **Random Forest** | плохо на sparse TF-IDF, огромная размерность, медленно, слабый signal |
| **Linear SVM** | силён на sparse vectors, но probabilities менее удобны |
| **Logistic Regression** | линейная модель на sparse features, быстрая, интерпретируемая, `predict_proba` из коробки |

**Вывод:** для baseline оптимален **TF-IDF + Logistic Regression**.

### 2.3. Почему multi-label, а не multiclass

Комментарий вроде *«You stupid disgusting idiot»* может быть одновременно:

- `toxic`
- `insult`
- `obscene`

Поэтому используется **multi-label binary classification**: отдельный бинарный классификатор на каждую метку через `OneVsRestClassifier`.

---

## 3. Архитектура Pipeline

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

### 3.1. Схема sklearn Pipeline

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

На inference backend вызывает только:

```python
model.predict_proba([text])  # → вероятности по 6 меткам
```

---

## 4. Файлы и функции проекта

### 4.1. Структура ML-кода

| Файл | Назначение |
|------|------------|
| `ml/preprocessing/text.py` | Функции препроцессинга текста |
| `ml/training/baseline_pipeline.py` | Сборка sklearn Pipeline |
| `ml/training/train_baseline.py` | CLI: загрузка данных, обучение, метрики, export |
| `ml/evaluation/metrics.py` | Multi-label метрики (F1, Hamming loss, per-label report) |
| `ml/experiments/baseline_tfidf_lr/metrics.json` | JSON с результатами hold-out оценки |
| `models/model.pkl` | Сериализованный Pipeline для FastAPI |
| `backend/app/services/inference.py` | Загрузка pickle и `predict_proba` в API |

### 4.2. `ml/preprocessing/text.py`

**`preprocess_text(text: str) -> str`**

Применяемые шаги:

1. HTML unescape (`&amp;` → `&`)
2. **lowercase**
3. удаление **URL** (`http://...`, `www....`)
4. удаление **HTML-тегов**
5. **нормализация пробелов**

**Что намеренно НЕ делаем:**

| Действие | Причина |
|----------|---------|
| Полное удаление пунктуации | `!!!` может быть сигналом агрессии |
| Удаление stopwords | «go to **hell**» — stopwords важны для токсичности |
| Aggressive stemming | можно потерять toxic-паттерны |

**`preprocess_batch(texts)`** — обёртка для sklearn `FunctionTransformer`, принимает iterable строк.

### 4.3. `ml/training/baseline_pipeline.py`

**`build_baseline_pipeline(...) -> Pipeline`**

Ключевые гиперпараметры по умолчанию:

| Компонент | Параметр | Значение | Смысл |
|-----------|----------|----------|-------|
| Word TF-IDF | `ngram_range` | `(1, 2)` | unigrams + bigrams |
| Word TF-IDF | `max_features` | `100_000` | ограничение словаря |
| Char TF-IDF | `analyzer` | `char_wb` | char n-grams с учётом границ слов |
| Char TF-IDF | `ngram_range` | `(3, 5)` | устойчивость к obfuscation |
| Оба TF-IDF | `min_df` | `5` | отсечение редкого шума |
| Оба TF-IDF | `max_df` | `0.95` | отсечение слишком частых токенов |
| Оба TF-IDF | `sublinear_tf` | `True` | `1 + log(tf)` — смягчает частые повторы |
| Logistic Regression | `solver` | `liblinear` | быстро и стабильно на sparse TF-IDF |
| Logistic Regression | `class_weight` | `balanced` | компенсация дисбаланса классов |
| Logistic Regression | `C` | `1.0` | стандартная L2-регуляризация |
| Logistic Regression | `max_iter` | `2000` | лимит итераций оптимизации |

**Метки (порядок фиксирован, совпадает с backend):**

`toxic`, `severe_toxic`, `obscene`, `threat`, `insult`, `identity_hate`

### 4.4. `ml/training/train_baseline.py`

**`load_jigsaw_dataset(path)`** — читает CSV, возвращает `(texts, y)` где `y` shape `(n_samples, 6)`.

**`train_and_evaluate(...)`** — train/test split, fit, hold-out метрики.

**CLI:**

```bash
# Полное обучение на Jigsaw train.csv
python -m ml.training.train_baseline --data data/raw/train.csv

# Быстрый smoke-test на demo-корпусе (без Kaggle-данных)
python -m ml.training.train_baseline --demo
```

Аргументы:

| Флаг | По умолчанию | Описание |
|------|--------------|----------|
| `--data` | `data/raw/train.csv` | Путь к CSV |
| `--out` | `models/model.pkl` | Куда сохранить модель |
| `--metrics-out` | `ml/experiments/baseline_tfidf_lr/metrics.json` | JSON с метриками |
| `--test-size` | `0.2` | Доля hold-out |
| `--threshold` | `0.5` | Порог для бинарных предсказаний при оценке |
| `--demo` | — | Принудительно demo-корпус |

### 4.5. `ml/evaluation/metrics.py`

**`multilabel_report(y_true, y_pred, label_names)`** возвращает:

- `hamming_loss`
- `f1_macro`, `f1_micro`
- `precision_macro`, `precision_micro`
- `recall_macro`, `recall_micro`
- `per_label` — precision/recall/F1/support по каждой метке
- `classification_report` — текстовый sklearn-отчёт

### 4.6. Интеграция с backend

`backend/app/services/inference.py`:

- **`ToxicInferenceService`** загружает `model.pkl` при старте FastAPI;
- **`predict_proba(text)`** возвращает `dict[label, float]`;
- **`_scores_from_predict_proba`** нормализует разные форматы sklearn-выхода (list vs ndarray).

API endpoint: `POST /api/predict` с телом `{"text": "..."}`.

---

## 5. Процесс обучения (фактический прогон)

### 5.1. Данные

| Параметр | Значение |
|----------|----------|
| Источник | `data/raw/train.csv` |
| Число комментариев | **159 571** |
| Train split | **127 656** (80%) |
| Test split | **31 915** (20%) |
| Stratify | по колонке `toxic` |
| Доля комментариев с ≥1 меткой | **~10.2%** |

### 5.2. Время обучения

На локальной машине (Windows, scikit-learn + liblinear): **~3.3 минуты** на полном датасете.

> Примечание: solver `saga` на этом объёме оказался слишком медленным; `liblinear` — стандартный и практичный выбор для sparse TF-IDF + OvR.

### 5.3. Порог и метрики

Для offline-оценки бинарные предсказания получаются порогом **0.5** по `predict_proba`.  
Для production/UI порог можно калибровать отдельно по каждой метке (особенно для редких классов).

---

## 6. Результаты hold-out evaluation (test 20%)

### 6.1. Общие метрики

| Метрика | Значение |
|---------|----------|
| **Hamming loss** | **0.0232** |
| **F1 macro** | **0.6162** |
| **F1 micro** | **0.7330** |
| Precision macro | 0.5160 |
| Precision micro | 0.6369 |
| Recall macro | 0.8025 |
| Recall micro | 0.8632 |

**Интерпретация:**

- **F1 micro ≈ 0.73** — хороший baseline на Jigsaw; модель уверенно ловит явную токсичность.
- **Recall > Precision** (особенно на редких классах) — модель склонна **перестраховываться** (много false positives), что типично при `class_weight="balanced"` и пороге 0.5.
- **Hamming loss ≈ 0.023** — в среднем ~2.3% меток ошибочны на комментарий.

### 6.2. Per-label F1

| Метка | Precision | Recall | F1 | Support (test) |
|-------|-----------|--------|-----|----------------|
| **toxic** | 0.717 | 0.873 | **0.787** | 3059 |
| **obscene** | 0.753 | 0.890 | **0.816** | 1710 |
| **insult** | 0.618 | 0.860 | **0.719** | 1590 |
| **identity_hate** | 0.338 | 0.768 | **0.469** | 289 |
| **severe_toxic** | 0.319 | 0.794 | **0.455** | 311 |
| **threat** | 0.351 | 0.629 | **0.450** | 97 |

**Наблюдения:**

- Лучше всего модель работает на **`obscene`** и **`toxic`** — много лексических маркеров.
- **`insult`** — уверенный средний результат (F1 ≈ 0.72).
- **`severe_toxic`**, **`threat`**, **`identity_hate`** — редкие классы с низкой precision; модель их находит (высокий recall), но часто «стреляет» ложно.

### 6.3. Качественные примеры (inference)

После обучения модель корректно отдаёт вероятности через backend-сервис:

- *«You are an idiot, shut up!»* → высокие scores для `toxic`, `insult`, `obscene`
- *«Thank you for the helpful article.»* → низкие scores по всем меткам

---

## 7. Как это работает end-to-end

```
1. Пользователь вводит текст в React UI (frontend/)
2. POST /api/predict → FastAPI (backend/)
3. ToxicInferenceService.predict_proba(text)
4. Pipeline:
     preprocess_batch → FeatureUnion(word+char TF-IDF) → 6× LogisticRegression
5. Ответ JSON: { "probabilities": { "toxic": 0.92, ... }, "labels": [...] }
6. UI рисует bar chart по каждой метке
```

Модель сериализуется через **pickle** — тот же объект Pipeline, что обучался в `ml/training/`, без отдельного conversion step.

---

## 8. Ограничения baseline и следующие шаги

### 8.1. Ограничения TF-IDF

TF-IDF **не понимает семантику**:

| Текст A | Текст B | TF-IDF |
|---------|---------|--------|
| «You are dumb» | «You are stupid» | разные векторы, хотя смысл один |

Не ловит:

- сарказм без toxic-лексики;
- завуалированную агрессию («I hope you have a *very* special day»);
- cross-lingual / code-switching без char-паттернов.

### 8.2. Возможные улучшения (Подход 2+)

1. **Калибровка порогов** per-label на validation set (повысит precision на редких классах).
2. **Linear SVM** как альтернативный linear baseline для сравнения.
3. **Embeddings / Transformers** (BERT, DistilBERT) — semantic understanding.
4. **Ensemble** TF-IDF + transformer scores.
5. **Cross-validation** вместо одного hold-out split для более стабильных метрик.

---

## 9. Быстрый старт (повторить эксперимент)

```bash
# из корня репозитория
python -m pip install -r ml/requirements.txt

# положить train.csv в data/raw/ (Kaggle Jigsaw)

python -m ml.training.train_baseline --data data/raw/train.csv

# запустить API
cd backend
python -m pip install -r requirements.txt
python -m uvicorn app.main:app --reload --host 127.0.0.1 --port 8000
```

---

## 10. Резюме

| Компонент | Решение |
|-----------|---------|
| Признаки | Word TF-IDF (1–2) + Char TF-IDF (3–5, `char_wb`) |
| Классификатор | `OneVsRestClassifier(LogisticRegression)` |
| Multi-label | 6 независимых бинарных LR |
| Препроцессинг | lowercase, URL/HTML removal, whitespace; пунктуация сохранена |
| Датасет | Jigsaw, 159 571 комментарий |
| Лучший F1 | `obscene` (0.82), `toxic` (0.79) |
| Общий F1 micro | **0.73** |
| Артефакт | `models/model.pkl` |

Baseline реализован, обучен на реальных данных, интегрирован с FastAPI backend и готов как **сильная отправная точка** перед embedding/transformer-подходами.
