"""Тесты evaluate() из model/train.py.

Модель подменена заглушкой с заранее заданными логитами: проверяем
арифметику метрик и работу порога, а не качество настоящей модели.
"""

import pytest
import torch

from model.train import evaluate
from tests.conftest import FakeModel

DEVICE = torch.device("cpu")


def run(logits, labels, threshold=0.5):
    """Гоняет evaluate на одном батче с заданными логитами и метками."""
    n = len(logits)
    batch = {
        "input_ids": torch.zeros((n, 4), dtype=torch.long),
        "attention_mask": torch.ones((n, 4), dtype=torch.long),
        "labels": torch.tensor(labels, dtype=torch.float32),
    }
    return evaluate(FakeModel(logits), [batch], DEVICE, threshold=threshold)


def test_perfect_predictions():
    result = run([5.0, -5.0, 5.0, -5.0], [1, 0, 1, 0])
    assert result["f1"] == 1.0
    assert result["precision"] == 1.0
    assert result["recall"] == 1.0


def test_all_spam_recall_is_one_precision_half():
    """Всё предсказано как spam: positives не пропущены, но есть ложные срабатывания."""
    result = run([5.0, 5.0, 5.0, 5.0], [1, 0, 1, 0])
    assert result["recall"] == 1.0
    assert result["precision"] == pytest.approx(0.5)


def test_all_spam_misses_negatives():
    result = run([5.0, 5.0, 5.0, 5.0], [1, 0, 1, 0])
    assert result["targets"].count(0) == 2
    assert sum(result["preds"]) == 4


def test_all_ham_precision_is_zero():
    """Ни одного положительного предсказания — precision не определён и равен нулю."""
    result = run([-5.0, -5.0, -5.0, -5.0], [1, 0, 1, 0])
    assert result["precision"] == 0.0
    assert result["f1"] == 0.0


def test_all_ham_misses_every_spam():
    """Ни одного положительного предсказания: precision и recall оба нулевые."""
    result = run([-5.0, -5.0, -5.0, -5.0], [1, 0, 1, 0])
    assert result["precision"] == 0.0
    assert result["recall"] == 0.0


def test_f1_matches_manual_computation():
    result = run([5.0, 5.0, 5.0, 5.0], [1, 0, 1, 0])
    # tp=2, fp=2, fn=0
    precision, recall = 2 / 4, 2 / 2
    expected = 2 * precision * recall / (precision + recall)
    assert result["f1"] == pytest.approx(expected)


def test_threshold_increases_predicted_positives():
    low = run([0.0, 0.0, 0.0, 0.0], [1, 0, 1, 0], threshold=0.4)
    high = run([0.0, 0.0, 0.0, 0.0], [1, 0, 1, 0], threshold=0.6)
    assert sum(low["preds"]) > sum(high["preds"])


def test_high_threshold_predicts_nothing():
    result = run([1.0, 2.0, 3.0, 4.0], [1, 0, 1, 0], threshold=0.99)
    assert sum(result["preds"]) == 0


def test_low_threshold_predicts_everything():
    result = run([-4.0, -3.0, -2.0, -1.0], [1, 0, 1, 0], threshold=0.01)
    assert sum(result["preds"]) == 4


def test_preds_and_targets_length_matches_batch():
    result = run([5.0, -5.0, 5.0, -5.0], [1, 0, 1, 0])
    assert len(result["preds"]) == 4
    assert len(result["targets"]) == 4


def test_targets_are_returned_unchanged():
    result = run([5.0, -5.0, 5.0, -5.0], [1, 0, 1, 0])
    assert list(result["targets"]) == [1, 0, 1, 0]


def test_metrics_in_unit_range():
    result = run([1.0, -1.0, 3.0, -3.0], [1, 0, 1, 0])
    for name in ("f1", "precision", "recall"):
        assert 0.0 <= result[name] <= 1.0


def test_predictions_are_numpy_values_not_tensors():
    """Возвращать нужно плоские значения: дальше они идут в CSV и classification_report."""
    result = run([5.0, -5.0, 5.0, -5.0], [1, 0, 1, 0])
    assert all(not hasattr(p, "requires_grad") for p in result["preds"])
    assert all(p in (0, 1, True, False) for p in result["preds"])


def test_multiple_batches_are_concatenated():
    """evaluate() должен склеивать батчи, а не брать только последний."""
    batches = []
    for _ in range(2):
        batches.append({
            "input_ids": torch.zeros((2, 4), dtype=torch.long),
            "attention_mask": torch.ones((2, 4), dtype=torch.long),
            "labels": torch.tensor([1, 0], dtype=torch.float32),
        })
    result = evaluate(FakeModel([5.0, -5.0]), batches, DEVICE, threshold=0.5)
    assert len(result["preds"]) == 4
    assert len(result["targets"]) == 4
    assert result["f1"] == 1.0
