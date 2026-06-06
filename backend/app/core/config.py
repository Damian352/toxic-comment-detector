"""
Backend configuration via environment variables and `.env`.

Default paths are resolved from the repository root (three levels up from this file).
In Docker, paths are overridden via env (`MODEL_PATH=/models/model.pkl`, etc.).
"""

from pathlib import Path

from pydantic_settings import BaseSettings, SettingsConfigDict

# Repository root: backend/app/core/config.py → parents[3]
_REPO_ROOT = Path(__file__).resolve().parents[3]
_DEFAULT_TFIDF_MODEL = _REPO_ROOT / "models" / "model.pkl"
_DEFAULT_BERT_DIR = _REPO_ROOT / "models" / "bert"
_DEFAULT_TFIDF_MODEL_PL = _REPO_ROOT / "models" / "model_pl.pkl"
_DEFAULT_BERT_DIR_PL = _REPO_ROOT / "models" / "bert_pl"
_DEFAULT_ML_EXPERIMENTS = _REPO_ROOT / "ml" / "experiments"
_DEFAULT_PROJECTIONS = _REPO_ROOT / "models" / "projections"


class Settings(BaseSettings):
    model_config = SettingsConfigDict(env_file=".env", env_file_encoding="utf-8", extra="ignore")

    # --- Inference artifacts (EN) ---
    model_path: Path = _DEFAULT_TFIDF_MODEL
    bert_model_dir: Path = _DEFAULT_BERT_DIR
    # --- Inference artifacts (PL) ---
    model_path_pl: Path = _DEFAULT_TFIDF_MODEL_PL
    bert_model_dir_pl: Path = _DEFAULT_BERT_DIR_PL
    # --- Training metrics for GET /api/metrics ---
    ml_experiments_dir: Path = _DEFAULT_ML_EXPERIMENTS
    projections_dir: Path = _DEFAULT_PROJECTIONS
    cors_origins: list[str] = ["http://localhost:5173", "http://127.0.0.1:5173"]
    # Minimum gcld3/langdetect confidence when lang=auto
    lang_detect_min_confidence: float = 0.70

settings = Settings()
