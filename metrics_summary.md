# Metryki projektu Toxic Comment Detector

## Model język angielski (Jigsaw Toxic Comment Classification)

### 1. TF-IDF + Logistic Regression (baseline_tfidf_lr)

**Globalne metryki (z dostrojonymi progami):**
| Metryka | Wartość |
|---------|---------|
| Hamming Loss | 0.0168 |
| **F1 Macro** | **0.6385** |
| **F1 Micro** | **0.7709** |
| Precision Macro | 0.6576 |
| Precision Micro | 0.7735 |
| Recall Macro | 0.6312 |
| Recall Micro | 0.7683 |

**Per-etykieta (F1, precision, recall, support):**
| Etykieta | Precision | Recall | F1 | Support |
|----------|-----------|--------|-----|---------|
| toxic | 0.8356 | 0.7957 | 0.8151 | 3059 |
| severe_toxic | 0.4478 | 0.5788 | 0.5049 | 311 |
| obscene | 0.8216 | 0.8322 | 0.8268 | 1710 |
| threat | 0.5323 | 0.3402 | 0.4151 | 97 |
| insult | 0.7307 | 0.7629 | 0.7465 | 1590 |
| identity_hate | 0.5774 | 0.4775 | 0.5227 | 289 |

**Progi dostrojone:** toxic=0.7, severe_toxic=0.9, obscene=0.7, threat=0.95, insult=0.75, identity_hate=0.9

---

### 2. BERT (bert_multilabel)

**Globalne metryki (z dostrojonymi progami):**
| Metryka | Wartość |
|---------|---------|
| Hamming Loss | 0.0146 |
| **F1 Macro** | **0.6933** (+0.055 vs baseline) |
| **F1 Micro** | **0.8068** (+0.036 vs baseline) |
| Precision Macro | 0.6981 |
| Precision Micro | 0.7861 |
| Recall Macro | 0.7033 |
| Recall Micro | 0.8287 |

**Per-etykieta (F1, precision, recall, support):**
| Etykieta | Precision | Recall | F1 | Support |
|----------|-----------|--------|-----|---------|
| toxic | 0.8336 | 0.8745 | 0.8535 | 3059 |
| severe_toxic | 0.4762 | 0.6109 | 0.5352 | 311 |
| obscene | 0.8418 | 0.8591 | 0.8504 | 1710 |
| threat | 0.6949 | 0.4227 | 0.5256 | 97 |
| insult | 0.7516 | 0.8088 | 0.7792 | 1590 |
| identity_hate | 0.5905 | 0.6436 | 0.6159 | 289 |

**Progi dostrojone:** toxic=0.45, severe_toxic=0.35, obscene=0.55, threat=0.6, insult=0.5, identity_hate=0.4

---

## Model język polski (BAN-PL Dataset)

### 3. TF-IDF + Logistic Regression (baseline_tfidf_lr_pl)

**Globalne metryki (z dostrojonymi progami):**
| Metryka | Wartość |
|---------|---------|
| Hamming Loss | 0.1257 |
| **F1 Macro** | **0.6173** |
| **F1 Micro** | **0.7493** |
| Precision Macro | 0.6273 |
| Precision Micro | 0.7475 |
| Recall Macro | 0.6119 |
| Recall Micro | 0.7511 |

**Per-etykieta (F1, precision, recall, support):**
| Etykieta | Precision | Recall | F1 | Support |
|----------|-----------|--------|-----|---------|
| safe | 0.8817 | 0.9145 | 0.8978 | 2397 |
| hate_speech | 0.6841 | 0.7363 | 0.7092 | 1350 |
| violence | 0.4474 | 0.3408 | 0.3869 | 537 |
| vulgarity | 0.4958 | 0.4561 | 0.4751 | 513 |

**Progi dostrojone:** safe=0.55, hate_speech=0.5, violence=0.65, vulgarity=0.65

---

### 4. HerBERT (bert_multilabel_pl)

**Globalne metryki (z dostrojonymi progami):**
| Metryka | Wartość |
|---------|---------|
| Hamming Loss | 0.1125 |
| **F1 Macro** | **0.6819** (+0.065 vs baseline) |
| **F1 Micro** | **0.7825** (+0.033 vs baseline) |
| Precision Macro | 0.6578 |
| Precision Micro | 0.7572 |
| Recall Macro | 0.7105 |
| Recall Micro | 0.8095 |

**Per-etykieta (F1, precision, recall, support):**
| Etykieta | Precision | Recall | F1 | Support |
|----------|-----------|--------|-----|---------|
| safe | 0.9276 | 0.9195 | 0.9235 | 2397 |
| hate_speech | 0.6934 | 0.8126 | 0.7483 | 1350 |
| violence | 0.4336 | 0.5233 | 0.4743 | 537 |
| vulgarity | 0.5766 | 0.5867 | 0.5816 | 513 |

**Progi dostrojone:** safe=0.45, hate_speech=0.4, violence=0.25, vulgarity=0.45

---

## Dlaczego zbieramy te metryki?

1. **Hamming Loss** — procent błędnie przewidzianych etykiet (niżej = lepiej)
2. **F1 Macro** — średnia F1 dla wszystkich etykiet (treats all labels equally, ważne dla niezbalansowanych danych)
3. **F1 Micro** — F1 liczona globalnie na wszystkich przykładach (ważna dla częstych etykiet)
4. **Precision/Recall** — kompromis między fałszywymi alarmami a pominięciami
5. **Per-label metrics** — szczegółowa analiza które etykiety są trudne (np. threat, violence)
6. **Baseline comparison** — porównanie progów 0.5 vs dostrojone per-etykieta
7. **Threshold tuning** — optymalne progi dla każdej etykiety (istotne dla rzadkich klas jak threat=0.6-0.95)
