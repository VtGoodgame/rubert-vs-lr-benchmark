"""Тесты маппинга меток: encode_labels и decode_labels."""

import numpy as np
import pandas as pd
import pytest

from model.data.labels import LABEL_MAP, decode_labels, encode_labels


def test_label_map():
    assert LABEL_MAP == {"ham": 0, "spam": 1}


def test_encode_labels_basic():
    df = pd.DataFrame({"target": ["ham", "spam", "ham"]})
    assert list(encode_labels(df)) == [0, 1, 0]


def test_encode_labels_is_case_and_space_insensitive():
    df = pd.DataFrame({"target": ["  HAM ", "Spam", "spam"]})
    assert list(encode_labels(df)) == [0, 1, 1]


def test_encode_labels_custom_column():
    df = pd.DataFrame({"label": ["spam", "ham"]})
    assert list(encode_labels(df, column="label")) == [1, 0]


def test_encode_labels_default_dtype_is_int8():
    df = pd.DataFrame({"target": ["spam", "ham"]})
    assert encode_labels(df).dtype == np.int8


def test_encode_labels_float32_dtype():
    df = pd.DataFrame({"target": ["spam", "ham"]})
    assert encode_labels(df, dtype="float32").dtype == np.float32


def test_encode_labels_unknown_raises():
    df = pd.DataFrame({"target": ["unknown"]})
    with pytest.raises(ValueError):
        encode_labels(df)


def test_encode_labels_mixed_known_and_unknown_raises():
    df = pd.DataFrame({"target": ["ham", "мусор"]})
    with pytest.raises(ValueError):
        encode_labels(df)


def test_encode_labels_empty_frame():
    assert list(encode_labels(pd.DataFrame({"target": []}))) == []


def test_decode_labels():
    assert decode_labels(np.array([0, 1, 0])) == ["ham", "spam", "ham"]


def test_decode_labels_accepts_list():
    assert decode_labels([1, 0]) == ["spam", "ham"]


def test_round_trip_encode_decode():
    df = pd.DataFrame({"target": ["ham", "spam", "spam", "ham"]})
    assert decode_labels(encode_labels(df, dtype="int64")) == list(df["target"])


def test_decode_labels_empty():
    assert decode_labels(np.array([])) == []
