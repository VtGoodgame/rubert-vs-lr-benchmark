"""Словарь меток. Живёт отдельно от dataset.py, чтобы скрипты на sklearn
не импортировали torch."""

import pandas as pd

LABEL_MAP = {"ham": 0, "spam": 1}


def encode_labels(frame: pd.DataFrame, column: str = "target", dtype: str = "int8"):
    """Переводит ham/spam в 0/1. Неизвестная метка — ошибка, а не молчаливый NaN."""
    mapped = frame[column].str.strip().str.lower().map(LABEL_MAP)
    if mapped.isna().any():
        bad = frame.loc[mapped.isna(), column].unique()
        raise ValueError(f"Неизвестные метки: {bad}")
    return mapped.values.astype(dtype)
