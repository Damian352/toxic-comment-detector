"""TF-IDF + Logistic Regression baseline pipeline for multi-label toxic detection."""

from __future__ import annotations

from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.multiclass import OneVsRestClassifier
from sklearn.pipeline import FeatureUnion, Pipeline
from sklearn.preprocessing import FunctionTransformer

from ml.labels import DEFAULT_THRESHOLD, LABELS
from ml.preprocessing.text import preprocess_batch


def build_baseline_pipeline(
    *,
    word_max_features: int = 100_000,
    char_max_features: int = 100_000,
    word_ngram_range: tuple[int, int] = (1, 2),
    char_ngram_range: tuple[int, int] = (3, 5),
    min_df: int = 5,
    max_df: float = 0.95,
    C: float = 1.0,
    max_iter: int = 2000,
    random_state: int = 42,
    preprocess_func=preprocess_batch,
) -> Pipeline:
    """
    Build a multi-label baseline: preprocess -> word TF-IDF + char TF-IDF -> OvR LR.

    Word n-grams capture explicit toxic vocabulary and short phrases ("shut up").
    Character n-grams capture obfuscated spellings ("1diot", "id!ot").
    """
    word_vectorizer = TfidfVectorizer(
        analyzer="word",
        ngram_range=word_ngram_range,
        min_df=min_df,
        max_df=max_df,
        max_features=word_max_features,
        sublinear_tf=True,
    )
    char_vectorizer = TfidfVectorizer(
        analyzer="char_wb",
        ngram_range=char_ngram_range,
        min_df=min_df,
        max_df=max_df,
        max_features=char_max_features,
        sublinear_tf=True,
    )
    classifier = OneVsRestClassifier(
        LogisticRegression(
            C=C,
            max_iter=max_iter,
            class_weight="balanced",
            solver="liblinear",
            random_state=random_state,
        ),
        n_jobs=1,
    )
    return Pipeline(
        [
            (
                "preprocess",
                FunctionTransformer(
                    preprocess_func,
                    validate=False,
                    feature_names_out="one-to-one",
                ),
            ),
            (
                "features",
                FeatureUnion(
                    [
                        ("word_tfidf", word_vectorizer),
                        ("char_tfidf", char_vectorizer),
                    ],
                ),
            ),
            ("clf", classifier),
        ]
    )
