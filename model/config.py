"""Пути и настройки слоя model."""

from pathlib import Path

from pydantic_settings import BaseSettings, SettingsConfigDict

BASE_DIR = Path(__file__).resolve().parent.parent
DATA_DIR = BASE_DIR / "data"
ARTIFACTS_DIR = BASE_DIR / "artifacts"


class Settings(BaseSettings):
    model_config = SettingsConfigDict(env_file=BASE_DIR / ".env", extra="ignore")

    train_file: str = "train.csv"
    test_file: str = "test.csv"
    text_column: str = "text"
    target_column: str = "target"
    model_path: Path = ARTIFACTS_DIR / "model.joblib"


settings = Settings()
