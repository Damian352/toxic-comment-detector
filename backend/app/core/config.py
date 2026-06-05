from pathlib import Path

from pydantic_settings import BaseSettings, SettingsConfigDict

_REPO_ROOT = Path(__file__).resolve().parents[3]
_DEFAULT_TFIDF_MODEL = _REPO_ROOT / "models" / "model.pkl"
_DEFAULT_BERT_DIR = _REPO_ROOT / "models" / "bert"


class Settings(BaseSettings):
    model_config = SettingsConfigDict(env_file=".env", env_file_encoding="utf-8", extra="ignore")

    model_path: Path = _DEFAULT_TFIDF_MODEL
    bert_model_dir: Path = _DEFAULT_BERT_DIR
    cors_origins: list[str] = ["http://localhost:5173", "http://127.0.0.1:5173"]


settings = Settings()
