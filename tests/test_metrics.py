import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from scripts.tracking.metrics import compute_all, compute_confusion, compute_metrics


def test_compute_confusion_perfect():
    preds = [1, 0, 1, 0]
    targets = [1, 0, 1, 0]
    conf = compute_confusion(preds, targets)
    assert conf == {"tp": 2, "fp": 0, "tn": 2, "fn": 0}


def test_compute_confusion_mixed():
    preds = [1, 1, 0, 0]
    targets = [1, 0, 0, 1]
    conf = compute_confusion(preds, targets)
    assert conf["tp"] == 1
    assert conf["fp"] == 1
    assert conf["tn"] == 1
    assert conf["fn"] == 1


def test_compute_metrics_all_correct():
    m = compute_metrics(tp=2, fp=0, tn=2, fn=0)
    assert m["precision"] == 1.0
    assert m["recall"] == 1.0
    assert m["f1"] == 1.0
    assert m["accuracy"] == 1.0


def test_compute_metrics_zero_division():
    m = compute_metrics(tp=0, fp=0, tn=0, fn=0)
    assert m["precision"] == 0.0
    assert m["recall"] == 0.0
    assert m["f1"] == 0.0
    assert m["accuracy"] == 0.0
    m = compute_metrics(tp=0, fp=5, tn=3, fn=0)
    assert m["precision"] == 0.0
    assert m["recall"] == 0.0
    assert m["f1"] == 0.0
    assert m["accuracy"] == 0.375


def test_compute_all():
    res = compute_all([1, 0, 1], [1, 1, 0])
    assert res["tp"] == 1
    assert res["fp"] == 1
    assert res["tn"] == 0
    assert res["fn"] == 1
    assert 0 <= res["f1"] <= 1.0