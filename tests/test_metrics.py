import pytest

from common.metrics import (
    DEFAULT_THRESHOLD_GRID,
    compute_all,
    compute_at_threshold,
    compute_confusion,
    compute_metrics,
    find_best_threshold,
)


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


def test_length_mismatch_raises():
    """Разная длина preds и targets — ошибка, а не молчаливое усечение.

    Без strict=True zip оборвался бы по короткому списку, и потерянные
    примеры тихо попали бы в метрики.
    """
    with pytest.raises(ValueError):
        compute_confusion([1, 0, 1], [1, 0])
    with pytest.raises(ValueError):
        compute_all([1, 0], [1, 0, 1])
def test_compute_at_threshold_uses_probabilities():
    """Порог решает судьбу вероятности: ниже — spam, выше или равно — ham."""
    probs = [0.9, 0.4]
    targets = [1, 0]

    assert compute_at_threshold(probs, targets, 0.5)["tp"] == 1
    assert compute_at_threshold(probs, targets, 0.2)["fp"] == 1


def test_compute_at_threshold_boundary_is_ham():
    """Равенство prob == threshold даёт ham: сетка порогов на этом и строит
    перебор, иначе верхний порог вёл бы себя как «всё spam»."""
    metrics = compute_at_threshold([0.5], [1], 0.5)
    assert metrics["tp"] == 0
    assert metrics["fn"] == 1


def test_find_best_threshold_picks_argmax():
    probs = [0.95, 0.92, 0.15, 0.10]
    targets = [1, 1, 0, 0]
    threshold, metrics = find_best_threshold(probs, targets)
    assert metrics["f1"] == 1.0
    assert threshold in DEFAULT_THRESHOLD_GRID


def test_find_best_threshold_is_reproducible_on_tie():
    """При равном f1 побеждает меньший порог, иначе результат зависел бы от
    порядка обхода сетки."""
    probs = [0.4, 0.45]
    targets = [1, 0]
    first, _ = find_best_threshold(probs, targets)
    second, _ = find_best_threshold(probs, targets)
    assert first == second


def test_find_best_threshold_rejects_empty_probs():
    with pytest.raises(ValueError, match="пуст"):
        find_best_threshold([], [])


def test_find_best_threshold_rejects_empty_grid():
    with pytest.raises(ValueError, match="пуста"):
        find_best_threshold([0.5], [1], grid=())
