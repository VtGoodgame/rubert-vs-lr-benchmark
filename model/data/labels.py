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


def decode_labels(y):
    import numpy as np
    y = np.asarray(y)
    result = []
    for val in y:
        result.append('spam' if int(val) == 1 else 'ham')
    return result

