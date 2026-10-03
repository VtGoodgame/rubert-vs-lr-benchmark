# tests/test_labels.py
import sys
from pathlib import Path

import pytest

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from model.data.labels import LABEL_MAP, decode_labels, encode_labels


def test_label_map():
    assert LABEL_MAP == {"ham": 0, "spam": 1}


def test_encode_labels_basic():
    import pandas as pd

    df = pd.DataFrame({"target": ["ham", "spam", "ham"]})
    y = encode_labels(df)
    assert list(y) == [0, 1, 0]


def test_encode_labels_unknown_raises():
    import pandas as pd

    df = pd.DataFrame({"target": ["unknown"]})
    with pytest.raises(ValueError):
        encode_labels(df)


def test_decode_labels():
    import numpy as np

    y = np.array([0, 1, 0])
    res = decode_labels(y)
    assert list(res) == ["ham", "spam", "ham"]