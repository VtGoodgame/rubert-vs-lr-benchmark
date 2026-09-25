"""Загрузка артефакта и предсказание. Единственная точка входа для service."""

from functools import lru_cache

import joblib
from sklearn.pipeline import Pipeline

from model.config import settings


@lru_cache(maxsize=1)
def load_model() -> Pipeline:
    if not settings.model_path.exists():
        raise FileNotFoundError(
            f"Артефакт не найден: {settings.model_path}. Запустите: python -m model.train"
        )
    return joblib.load(settings.model_path)


def predict(texts: list[str]) -> list[str]:
    return [str(label) for label in load_model().predict(texts)]
