"""Словарь меток. Живёт отдельно от dataset.py, чтобы скрипты на sklearn
не импортировали torch."""

import pandas as pd

from common.config import LABELS

# Порядок берётся из common.config: нулевой код соответствует первому имени
# в LABELS. Словарь в одном месте нужен, чтобы «spam» не оказался 0 в одном
# слое и 1 в другом.
LABEL_MAP = {name: code for code, name in enumerate(LABELS)}


def encode_labels(frame: pd.DataFrame, column: str = "target", dtype: str = "int8"):
    """Переводит ham/spam в 0/1. Неизвестная метка — ошибка, а не молчаливый NaN.

    astype(str) обязателен: на пустом фрейме pandas выводит тип float64,
    и .str без него падает с AttributeError вместо внятной ошибки про метки.
    """
    mapped = frame[column].astype(str).str.strip().str.lower().map(LABEL_MAP)
    if mapped.isna().any():
        bad = frame.loc[mapped.isna(), column].unique()
        raise ValueError(f"Неизвестные метки: {bad}")
    return mapped.values.astype(dtype)


def decode_labels(y):
    """Переводит коды обратно в ham/spam — тем же порядком, что задаёт LABEL_MAP."""
    import numpy as np

    return [LABELS[int(val)] for val in np.asarray(y)]

