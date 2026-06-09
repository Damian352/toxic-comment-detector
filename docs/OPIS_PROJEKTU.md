# Toxic Comment Detector — opis projektu

Dokument opisuje system wykrywania toksycznych komentarzy: od danych treningowych, przez pipeline ML, po interfejs użytkownika i plany wdrożenia produkcyjnego.

---

## 1. Temat projektu i zgodność z wymaganiami

Projekt **Toxic Comment Detector** to system analizy tekstu, który automatycznie wykrywa toksyczne komentarze w języku naturalnym. Problem jest sformułowany jako **klasyfikacja wieloetykietowa** — jeden komentarz może jednocześnie należeć do kilku kategorii (np. `toxic` + `insult` + `obscene`). System obsługuje **język angielski** (dane Jigsaw) oraz **polski** (dane BAN-PL) i może służyć do moderacji treści na forach, portalach społecznościowych lub platformach edukacyjnych.

Zakres z opisu tematu został zrealizowany w praktyce: jest **przetwarzanie tekstu** (czyszczenie, normalizacja), **reprezentacja TF-IDF i embeddingów transformerowych**, **modele klasyfikacyjne** (Logistic Regression i BERT/HerBERT), **interfejs użytkownika** z prawdopodobieństwami per klasa oraz **ewaluacja** na zbiorze testowym. Dodatkowo projekt wykracza poza minimalny opis tematu — oferuje tryb porównania dwóch modeli, mapy semantyczne PCA i automatyczne wykrywanie języka.

Wymagania akademickie projektu są spełnione:

| Wymaganie | Status | Realizacja w projekcie |
|-----------|--------|------------------------|
| Komponent NLP (najlepiej >1) | ✅ | Klasyfikacja wieloetykietowa, embeddingi TF-IDF i BERT/HerBERT, wykrywanie języka (gcld3/langdetect) |
| Działający system (nie tylko eksperyment) | ✅ | Backend FastAPI + frontend React + Docker Compose |
| Interfejs użytkownika | ✅ | Panel analizy, wykresy, dashboard metryk, mapy 2D/3D |
| Ewaluacja (metryka + analiza) | ✅ | F1 macro/micro, precision/recall per klasa, tuning progów, analiza błędów na mapach PCA |

Brakuje natomiast komponentów takich jak **NER** czy **LLM** — nie były one wymagane explicite, ale mogłyby wzbogacić projekt w przyszłości.

---

## 2. Zbiory danych

### 2.1. Jigsaw Toxic Comment Classification (angielski)

Główny zbiór angielski pochodzi z konkursu Kaggle [Jigsaw Toxic Comment Classification Challenge](https://www.kaggle.com/c/jigsaw-toxic-comment-classification-challenge). Plik `data/raw/train.csv` zawiera **159 571** komentarzy z Wikipedii, każdy opatrzony sześcioma etykietami binarnymi: `toxic`, `severe_toxic`, `obscene`, `threat`, `insult`, `identity_hate`.

Zbiór jest **silnie niezbalansowany**. Najczęstsza etykieta `toxic` występuje w ok. 9,6% wierszy (~15 294 przykładów), podczas gdy `threat` tylko w 0,30% (~478) i `identity_hate` w 0,88% (~1 405). Wiele komentarzy ma **więcej niż jedną etykietę** — typowe jest łączenie `toxic` z `insult` lub `obscene`. Analiza EDA w notebooku `ml/notebooks/01_eda.ipynb` pokazuje też rozkład długości tekstów: mediana to ok. 60 znaków, ale ogon rozkładu sięga kilku tysięcy znaków.

Wizualizacje EDA (matplotlib/seaborn) obejmują: **słupkowy wykres liczności klas**, **histogram długości komentarzy** (pełny i obcięty do percentyla 99), **rozkład liczby etykiet na komentarz** oraz **macierz współwystępowania etykiet** (heatmapa). Wykresy te pomagają zrozumieć, dlaczego rzadkie klasy (`threat`, `severe_toxic`) są trudniejsze do wykrycia i dlaczego stosuje się `class_weight="balanced"` w modelu bazowym.

### 2.2. BAN-PL (polski)

Polski zbiór **BAN-PL** pochodzi z komentarzy serwisu Wykop.pl i jest przechowywany w `BAN-PL_2/BAN-PL.csv` (nie jest dystrybuowany w repozytorium). Zawiera ok. **24 000** komentarzy z czterema etykietami: `safe`, `hate_speech`, `violence`, `vulgarity`. Semantyka różni się od Jigsaw — klasa `safe` oznacza brak naruszeń, a pozostałe trzy opisują konkretne typy szkodliwości.

Polski korpus ma inną strukturę etykiet (w tym jawna klasa „bezpieczny”) i często zawiera placeholdery typu `{USERNAME}`, `{URL}`, które są usuwane w preprocessing. Zbiór testowy (20% hold-out) liczy **4 797** przykładów. Klasa `safe` dominuje (~50% testu), podczas gdy `violence` i `vulgarity` są rzadsze — stąd niższe wyniki F1 dla tych kategorii w modelu TF-IDF.

---

## 3. Pipeline systemu

Pipeline można podzielić na część **offline** (trening i ewaluacja) oraz **online** (inferencja przez API i UI).

### 3.1. Pipeline offline (trening)

```
Dane CSV → preprocessing → podział train/test (80/20, seed=42)
    → trening modelu (TF-IDF+LR lub BERT/HerBERT)
    → predykcje na zbiorze testowym
    → tuning progów decyzyjnych per etykieta
    → zapis metryk (ml/experiments/*/metrics.json)
    → opcjonalnie: mapy PCA (models/projections/)
```

Kroki uruchamiane z katalogu głównego repozytorium:

```bash
python -m ml.training.train_baseline          # EN: TF-IDF → models/model.pkl
python -m ml.training.train_bert --epochs 3   # EN: BERT → models/bert/
python -m ml.training.train_baseline_pl       # PL: TF-IDF → models/model_pl.pkl
python -m ml.training.train_bert_pl --epochs 3  # PL: HerBERT → models/bert_pl/
python -m ml.training.tune_thresholds         # progi → models/thresholds.json
python -m ml.training.build_projection_maps   # mapy PCA → models/projections/
```

### 3.2. Pipeline online (inferencja)

```
Tekst użytkownika → wykrycie języka (auto/en/pl)
    → preprocessing (EN lub PL)
    → inferencja wybranego modelu (tfidf_lr / bert / both)
    → prawdopodobieństwa sigmoid per etykieta
    → progi z thresholds.json → aktywne etykiety
    → opcjonalnie: mapa PCA lub mapa kotwic referencyjnych
    → odpowiedź JSON → wizualizacja w React
```

Endpoint główny to `POST /api/predict`. Parametr `include_pca: false` (domyślnie) przyspiesza odpowiedź — wtedy zwracane są tylko prawdopodobieństwa i mapa kotwic. Parametr `model: both` uruchamia oba modele równolegle do porównania w UI.

### 3.3. Preprocessing tekstu

Dla angielskiego (`preprocess_text`): odkodowanie HTML, małe litery, usunięcie URL-i i tagów HTML, normalizacja białych znaków. Dla polskiego (`preprocess_text_pl`): usunięcie `{USERNAME}`, `{URL}`, `&gt;`, URL-i, normalizacja spacji — **bez** agresywnej lematyzacji, aby zachować sygnały wulgarności i formy słów.

---

## 4. Metody wykrywania toksyczności

### 4.1. TF-IDF + Logistic Regression (baseline)

Pierwsza metoda to klasyczny pipeline sklearn z pliku `ml/training/baseline_pipeline.py`:

1. **Word TF-IDF** — n-gramy 1–2 (słowa i krótkie frazy typu „shut up”).
2. **Char TF-IDF** — n-gramy znaków 3–5 z analizatorem `char_wb` (łapie obfuskację: `id!ot`, `f*ck`).
3. **OneVsRestClassifier(LogisticRegression)** — osobny klasyfikator binarny na każdą etykietę.

**Dlaczego ta metoda?** Toksyczność często przejawia się przez konkretne słowa i wzorce znaków. TF-IDF dobrze opisuje krótkie teksty, trenuje się szybko (sekundy–minuty), daje `predict_proba` i jest łatwy do wdrożenia w produkcji przy niskim koszcie obliczeniowym. Regresja logistyczna dobrze współpracuje ze **sparse** wektorami wysokiej wymiarowości.

### 4.2. BERT / HerBERT (transformer)

Druga metoda wykorzystuje modele transformerowe z Hugging Face:

| Język | Model bazowy | Artefakt |
|-------|-------------|----------|
| EN | `bert-base-uncased` | `models/bert/` |
| PL | `allegro/herbert-base-cased` | `models/bert_pl/` |

Architektura: `AutoModelForSequenceClassification` z `problem_type="multi_label_classification"`. Na wyjściu **sigmoid** na każdym logicie (nie softmax — etykiety są niezależne). Tokenizacja WordPiece, maks. długość 256 tokenów, embedding `[CLS]` reprezentuje cały komentarz.

**Dlaczego BERT?** Kontekst i znaczenie słów w zdaniu — model lepiej rozumie ironię, złożone groźby i subtelne obrazy. Na Jigsaw BERT osiąga wyższy F1 macro (~0,69 vs ~0,64 dla TF-IDF). Koszt: wolniejsza inferencja, większe zużycie GPU/RAM, wyższy koszt w modelu billingowym produkcyjnym.

### 4.3. Wykrywanie języka i routing modeli

Moduł `backend/app/services/lang_detect.py` kieruje tekst do właściwego zestawu modeli (EN lub PL). W Dockerze używany jest **gcld3**, lokalnie fallback **langdetect**. Użytkownik może też wymusić język (`lang: en` / `pl`). Dzięki temu jeden endpoint obsługuje oba języki bez ręcznego wyboru przez moderatora.

### 4.4. Progi decyzyjne

Domyślny próg 0,5 jest zbyt sztywny dla niezbalansowanych klas. Skrypt `tune_thresholds.py` szuka optymalnego progu per etykieta (siatka 0,05–0,95) na zbiorze walidacyjnym, maksymalizując F1. Wyniki trafiają do `models/thresholds.json` — **bez ponownego trenowania** wystarczy zmienić progi. Np. dla `threat` (EN) progi są wysokie (0,95 TF-IDF, 0,6 BERT), bo fałszywe alarmy są kosztowne.

---

## 5. Metryki ewaluacji i wyniki

### 5.1. Wybrane metryki i uzasadnienie

Funkcja `multilabel_report()` w `ml/evaluation/metrics.py` oblicza:

| Metryka | Znaczenie | Dlaczego używana |
|---------|-----------|------------------|
| **F1 macro** | Średnia F1 po klasach (równe wagi) | Pokazuje jakość na **rzadkich** klasach (`threat`, `identity_hate`) |
| **F1 micro** | F1 globalnie po wszystkich predykcjach | Odzwierciedla ogólną skuteczność przy dominujących etykietach |
| **Precision / Recall** per klasa | Dokładność i czułość dla każdej etykiety | Pozwala ocenić kompromis FP vs FN per kategoria |
| **Hamming loss** | Ułamek błędnie przypisanych etykiet | Naturalna metryka dla multi-label; im niższa, tym lepiej |

Dla zadania moderacji **precision** jest często ważniejsza niż recall (fałszywe oskarżenie użytkownika jest kosztowne), ale zbyt wysoka precision przy niskim recall oznacza przepuszczanie szkodliwych treści — stąd tuning progów i analiza per klasa.

### 5.2. Wyniki na zbiorze testowym (hold-out 20%)

Dane z `ml/experiments/*/metrics.json` (po tuningu progów):

#### Angielski (Jigsaw, n_test = 31 915)

| Model | F1 macro | F1 micro | Hamming loss |
|-------|----------|----------|--------------|
| TF-IDF + LR | **0,639** | 0,771 | 0,0168 |
| BERT | **0,693** | 0,807 | 0,0146 |

**Komentarz:** BERT wygrywa o ~5 punktów procentowych F1 macro. Obie metody dobrze radzą sobie z `toxic`, `obscene`, `insult` (F1 > 0,75). Najtrudniejsza klasa to **`threat`** (F1 ≈ 0,42 TF-IDF, 0,53 BERT) — mało przykładów treningowych i subtelny język. `severe_toxic` i `identity_hate` są średnie (F1 0,50–0,62). Tuning progów poprawił F1 macro TF-IDF z 0,616 (przy 0,5) do 0,639.

#### Polski (BAN-PL, n_test = 4 797)

| Model | F1 macro | F1 micro | Hamming loss |
|-------|----------|----------|--------------|
| TF-IDF + LR | **0,617** | 0,749 | 0,126 |
| HerBERT | **0,682** | 0,782 | 0,113 |

**Komentarz:** HerBERT wyraźnie lepszy, szczególnie na `hate_speech` (F1 0,75 vs 0,71) i `safe` (F1 0,92). `violence` i `vulgarity` pozostają słabsze (F1 < 0,58) — mała liczba przykładów i nakładanie się kategorii. Hamming loss jest wyższy niż dla EN, bo polski zbiór ma inną semantykę etykiet (w tym jawna klasa `safe`).

### 5.3. Analiza jakościowa

Poza metrykami liczbowymi projekt oferuje **analizę błędów** na mapach PCA: każdy punkt walidacyjny ma typ `correct`, `false_positive`, `false_negative` lub `label_mismatch`. Filtry w UI pozwalają przeglądać tylko błędne klasyfikacje i czytać treść komentarza w tooltipie. Tryb **dual comparison** (`model: both`) umożliwia bezpośrednie porównanie TF-IDF vs BERT na tym samym tekście — widać, że BERT lepiej łapie kontekstowe groźby, a TF-IDF czasem reaguje na pojedyncze słowa kluczowe.

---

## 6. Wizualizacja danych

### 6.1. EDA (notebook)

W `ml/notebooks/01_eda.ipynb` (matplotlib, seaborn, pandas):

- Słupkowy wykres **liczności etykiet** Jigsaw.
- Histogram **długości komentarzy** (znaki) — pełny i obcięty.
- Wykres **liczby etykiet na komentarz** (multi-label).
- Heatmapa **współwystępowania etykiet**.

### 6.2. Interfejs użytkownika (React)

Frontend (`frontend/src/App.tsx`) oferuje:

| Wizualizacja | Technologia | Opis |
|--------------|-------------|------|
| **Wykres radarowy** | SVG (custom) | Prawdopodobieństwa per etykieta; w trybie dual dwa wielokąty (TF-IDF vs BERT) |
| **Mapa PCA walidacji** | SVG 2D / Plotly 3D | Chmura punktów ze zbioru testowego; kolory = typ błędu |
| **Mapa kotwic referencyjnych** | SVG 2D | Stałe komentarze wzorcowe; pozycja aktywnego tekstu z profilu prawdopodobieństw |
| **Dashboard metryk** | SVG grouped bar chart | F1, precision, recall per klasa dla 4 modeli |
| **Tooltips interaktywne** | React state | Szczegóły punktu po najechaniu (tekst, etykiety, błąd) |

Mapa PCA jest budowana offline (`build_projection_maps.py`) z embeddingów TF-IDF (TruncatedSVD → PCA) lub BERT (`[CLS]` → PCA). Mapa kotwic działa zawsze — nie wymaga dodatkowych obliczeń embeddingowych przy inferencji.

---

## 7. Struktura projektu

```
toxic-comment-detector/
├── ml/                      # Trening, ewaluacja, wizualizacja ML
│   ├── labels.py            # Taksonomie EN/PL, progi
│   ├── preprocessing/       # Czyszczenie tekstu EN/PL
│   ├── training/            # Skrypty treningu, tuning, mapy PCA
│   ├── evaluation/          # Metryki, tuning progów
│   ├── visualization/       # Embeddingi i redukcja wymiarów
│   ├── experiments/         # metrics.json per model
│   ├── notebooks/           # EDA (01_eda.ipynb)
│   └── reports/             # Raporty etapowe
├── backend/                 # FastAPI — inferencja REST API
│   └── app/
│       ├── api/routes.py    # Endpointy /predict, /metrics, ...
│       ├── services/        # Inference, BERT, projekcje, język
│       └── core/config.py   # Ścieżki modeli, zmienne środowiskowe
├── frontend/                # React + Vite — UI analizy
│   └── src/App.tsx          # Główny panel (wykresy, mapy, metryki)
├── models/                  # Artefakty (pickle, wagi HF, progi, projekcje)
├── data/raw/                # train.csv Jigsaw (poza git)
├── BAN-PL_2/                # BAN-PL.csv (poza git)
├── docs/                    # Dokumentacja (BACKEND.md, ML.md, ten plik)
├── images/demo/             # Zrzuty ekranu UI
└── docker-compose.yml       # Uruchomienie backend + frontend
```

Podział odpowiedzialności: **`ml/`** trenuje i ocenia, **`backend/`** serwuje modele przez API, **`frontend/`** prezentuje wyniki. Wspólne stałe (`LABELS`, `PL_LABELS`, progi) żyją w `ml/labels.py` i są importowane zarówno w treningu, jak i w inferencji.

---

## 8. Najważniejsze fragmenty kodu

### 8.1. Taksonomia etykiet i progi (`ml/labels.py`)

Centralne miejsce definicji etykiet EN (6 klas toksyczności) i PL (`safe` + 3 naruszenia). Funkcje `get_per_label_thresholds()` i `active_labels_from_probs()` tłumaczą surowe prawdopodobieństwa na etykiety widoczne w UI — z osobną logiką dla polskiego (priorytet naruszeń nad `safe`).

### 8.2. Pipeline baseline (`ml/training/baseline_pipeline.py`)

Buduje sklearn `Pipeline`: preprocess → `FeatureUnion`(word TF-IDF + char TF-IDF) → `OneVsRestClassifier(LR)`. To serce modelu klasycznego — prosty, szybki, interpretowalny.

### 8.3. Metryki (`ml/evaluation/metrics.py`)

Funkcja `multilabel_report()` — jeden punkt wejścia do raportowania F1 macro/micro, Hamming loss i metryk per etykieta. Używana w treningu, tuningu progów i eksporcie do `metrics.json`.

### 8.4. Rejestr inferencji (`backend/app/services/registry.py`)

Klasa `InferenceRegistry` ładuje cztery modele (TF-IDF EN/PL, BERT EN/PL), śledzi flagi `loaded` i kieruje `predict_proba(text)` do właściwego backendu. Obsługuje tryb `both` dla porównania.

### 8.5. API (`backend/app/api/routes.py`)

Endpoint `POST /api/predict` — główna brama systemu. Przyjmuje tekst, model, język i flagę PCA; zwraca prawdopodobieństwa, aktywne etykiety i opcjonalnie projekcje 2D/3D. Endpoint `GET /api/metrics` zasila dashboard w frontendzie.

### 8.6. UI (`frontend/src/App.tsx`)

Monolityczny komponent (~2500 linii) łączący formularz analizy, wykres radarowy, mapy semantyczne, filtry błędów i panel metryk. Obsługuje UI w języku polskim i angielskim.

---

## 9. Stan projektu i wdrożenie produkcyjne

### 9.1. Obecny stan

System jest **w pełni funkcjonalny jako prototyp / MVP**:

- ✅ Trenowanie i ewaluacja 4 modeli (EN/PL × TF-IDF/BERT).
- ✅ REST API z dokumentacją OpenAPI (`/docs`).
- ✅ Interfejs webowy z wizualizacjami i porównaniem modeli.
- ✅ Docker Compose do szybkiego uruchomienia.
- ✅ Metryki hold-out, tuning progów, mapy błędów.

**Czego brakuje do produkcji:**

- Brak **autentykacji** i **autoryzacji** (API jest otwarte).
- Brak **rate limiting**, ochrony przed nadużyciami (DDoS, prompt injection na poziomie API).
- Brak **systemu kont**, billing i kluczy API z limitami tokenów.
- Brak **monitoringu** (metryki runtime, alerty degradacji modelu).
- Brak **CI/CD**, testów integracyjnych i load testów.
- Brak **skalowania** (jedna instancja, modele w pamięci procesu).
- Brak **wersjonowania modeli** i A/B testów w produkcji.

### 9.2. Plan wdrożenia produkcyjnego

Proponowana architektura SaaS:

```
Klient → API Gateway (auth, rate limit, billing)
    → Router modeli (TF-IDF tani / BERT drogi)
    → Worker pool inferencji (CPU dla TF-IDF, GPU dla BERT)
    → PostgreSQL (konta, klucze, zużycie tokenów)
    → Redis (cache, limity, kolejka)
```

**System kont i billing (propozycja użytkownika):**

- Rejestracja / logowanie (OAuth2 lub e-mail + hasło).
- Per konto: plan, saldo, historia zużycia.
- **Klucze API** z limitami (np. 10 000 zapytań/miesiąc, 500 000 tokenów).
- **Cennik zróżnicowany:** TF-IDF — np. 1 jednostka za 1000 znaków; BERT — np. 10 jednostek za 1000 znaków (ze względu na GPU i latencję).
- Panel do tworzenia / unieważniania kluczy, podgląd metryk zużycia.

**Ochrona przed atakami:**

- Rate limiting per klucz IP / API key (np. token bucket w Redis).
- Walidacja wejścia (max 8000 znaków — już jest), sanityzacja, timeout inferencji.
- WAF / reverse proxy (Cloudflare, nginx) przed API.
- Izolacja procesów inferencji (kontenery, limity CPU/RAM).
- Logowanie i alerting przy anomaliach (np. 1000 req/s z jednego klucza).
- Opcjonalnie: kolejka zadań dla BERT przy dużym obciążeniu.

### 9.3. Szacowany czas prac

| Etap | Zakres | Czas (1 dev) |
|------|--------|--------------|
| Auth + konta użytkowników | JWT/OAuth, CRUD użytkowników, PostgreSQL | 2–3 tygodnie |
| Klucze API + limity tokenów | Generowanie kluczy, middleware, Redis counters | 1–2 tygodnie |
| Billing (podstawowy) | Plany, meterowanie, prosty cennik TF-IDF/BERT | 2–3 tygodnie |
| Bezpieczeństwo | Rate limit, WAF, hardening Docker, HTTPS | 1–2 tygodnie |
| Infrastruktura prod | K8s lub managed service, GPU node, CI/CD | 2–4 tygodnie |
| Monitoring i testy | Prometheus/Grafana, testy obciążeniowe, SLA | 1–2 tygodnie |
| **Razem (MVP produkcyjny)** | | **~9–16 tygodni** |

Przy zespole 2–3 osób i gotowych usługach chmurowych (Auth0, Stripe) można skrócić do **6–10 tygodni**.

### 9.4. Kierunki rozwoju (kierunki rozwoju)

1. **Lepsze modele** — fine-tuning na domenie klienta, ensemble TF-IDF + BERT, modele wielojęzyczne (XLM-R).
2. **Więcej funkcji NLP** — NER do wykrywania celów ataków, embeddingi do wyszukiwania podobnych komentarzy, opcjonalnie LLM do wyjaśnień decyzji.
3. **Aktywne uczenie** — moderator oznacza błędy → automatyczny retrening i drift detection.
4. **Explainability** — LIME/SHAP dla TF-IDF, attention maps dla BERT.
5. **Integracje** — webhooki, pluginy do Discourse/WordPress, batch API dla archiwów.
6. **Rozszerzenie języków** — ukraiński, niemiecki, korpora domenowe.
7. **Edge deployment** — skwantyzowany model ONNX dla TF-IDF na urządzeniach brzegowych.

---

## 10. Podsumowanie

**Toxic Comment Detector** to kompletny system klasyfikacji wieloetykietowej toksycznych komentarzy w języku angielskim i polskim. Łączy szybki baseline TF-IDF+LR z dokładniejszym BERT/HerBERT, oferuje interfejs do analizy i porównania modeli oraz pipeline ewaluacji z metrykami, tuningiem progów i wizualną analizą błędów.

Wymagania projektu akademickiego (NLP, działający system, UI, ewaluacja) są **spełnione**. Temat z opisu projektu (preprocessing, TF-IDF/embeddingi, klasyfikacja wieloetykietowa, UI z prawdopodobieństwami, F1 macro/micro, analiza błędów, porównanie modeli) jest **zrealizowany w całości**, z dodatkowymi rozszerzeniami (PL, mapy PCA, auto-detekcja języka).

Do wdrożenia produkcyjnego potrzebne są przede wszystkim: **konta użytkowników, klucze API z limitami, billing zróżnicowany per model oraz warstwa bezpieczeństwa** — szacunkowo 2–4 miesiące pracy jednego doświadczonego developera.

---

*Dokument wygenerowany na podstawie stanu repozytorium: czerwiec 2026.*
