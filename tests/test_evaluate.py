"""Тесты evaluate() и save_artifacts() из model/train.py.

Модель подменена заглушкой с заранее заданными логитами: проверяем арифметику
метрик и работу порога, а не качество настоящей модели.

evaluate() метрики не считает — это делает common.metrics, чтобы обе модели
считались одним кодом. Поэтому здесь метрики получаются через compute_all,
ровно как в model/train.py.
"""

import pytest
import torch

from common.config import METRIC_NAMES
from common.metrics import compute_all
from model.train import evaluate, save_artifacts
from tests.conftest import FakeModel

# Маркер снимает эти тесты в CI: они проверяют код обучения, а не независимые
# функции, и на гонки в GPU не рассчитаны. Локально `pytest` без -m гоняет их.
pytestmark = pytest.mark.torch

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


def scored(logits, labels, threshold=0.5):
    """evaluate + метрики тем же кодом, что и в обучении."""
    result = run(logits, labels, threshold)
    return compute_all(result["preds"], result["targets"])


def test_perfect_predictions():
    result = scored([5.0, -5.0, 5.0, -5.0], [1, 0, 1, 0])
    assert result["f1"] == 1.0
    assert result["precision"] == 1.0
    assert result["recall"] == 1.0


def test_all_spam_recall_is_one_precision_half():
    """Всё предсказано как spam: positives не пропущены, но есть ложные срабатывания."""
    result = scored([5.0, 5.0, 5.0, 5.0], [1, 0, 1, 0])
    assert result["recall"] == 1.0
    assert result["precision"] == pytest.approx(0.5)


def test_all_spam_misses_negatives():
    result = run([5.0, 5.0, 5.0, 5.0], [1, 0, 1, 0])
    assert result["targets"].count(0) == 2
    assert sum(result["preds"]) == 4


def test_all_ham_precision_is_zero():
    """Ни одного положительного предсказания — precision не определён и равен нулю."""
    result = scored([-5.0, -5.0, -5.0, -5.0], [1, 0, 1, 0])
    assert result["precision"] == 0.0
    assert result["f1"] == 0.0


def test_all_ham_misses_every_spam():
    """Ни одного положительного предсказания: precision и recall оба нулевые."""
    result = scored([-5.0, -5.0, -5.0, -5.0], [1, 0, 1, 0])
    assert result["precision"] == 0.0
    assert result["recall"] == 0.0


def test_f1_matches_manual_computation():
    result = scored([5.0, 5.0, 5.0, 5.0], [1, 0, 1, 0])
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
    result = scored([1.0, -1.0, 3.0, -3.0], [1, 0, 1, 0])
    for name in METRIC_NAMES:
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
    assert compute_all(result["preds"], result["targets"])["f1"] == 1.0


def test_evaluate_returns_no_metrics():
    """Метрики считает common.metrics. Если evaluate начнёт считать свои,
    у моделей разойдутся числа на вырожденных данных — именно из-за этого его
    метрики и вынесены."""
    result = run([5.0, -5.0, 5.0, -5.0], [1, 0, 1, 0])
    assert set(result) == {"preds", "probs", "targets"}


def test_evaluate_returns_probabilities():
    """Порог подбирается по val, а для этого нужны вероятности, а не метки."""
    result = run([5.0, -5.0], [1, 0])
    assert len(result["probs"]) == 2
    assert result["probs"][0] > result["probs"][1]
    assert 0.0 < result["probs"][0] < 1.0


def test_probabilities_follow_sigmoid():
    """Проверяем именно transform: prob растёт с логитом."""
    low = run([0.0], [0])["probs"][0]
    high = run([3.0], [0])["probs"][0]
    assert low == pytest.approx(0.5, abs=1e-6)
    assert high > low


def test_metrics_include_accuracy():
    """accuracy обязана быть: без неё в rubert_metrics.csv попадали три метрики
    против четырёх у baseline, графики различались по длине, а compare.py
    падал с «в файле нет метрик: accuracy»."""
    result = scored([5.0, -5.0, 5.0, -5.0], [1, 0, 1, 0])
    assert set(METRIC_NAMES) <= set(result)
    assert result["accuracy"] == 1.0


def test_save_artifacts_writes_every_metric(tmp_path, monkeypatch):
    """CSV RuBERT обязан содержать те же метрики, что и CSV baseline: иначе
    сравнивать пришлось бы вслепую."""
    import pandas as pd

    captured = {}
    monkeypatch.setattr(
        "model.train.plot_test_comparison",
        lambda chart, **kw: captured.update(chart=chart, path=kw.get("save_path")),
    )

    result = run([5.0, -5.0, 5.0, 5.0], [1, 0, 1, 0])
    metrics = compute_all(result["preds"], result["targets"])

    csv_path = tmp_path / "rubert_metrics.csv"
    plot_path = tmp_path / "rubert_metrics.png"
    save_artifacts(metrics, 0.63, csv_path=csv_path, plot_path=plot_path)

    row = pd.read_csv(csv_path).iloc[0]
    for name in METRIC_NAMES:
        assert row[name] == pytest.approx(metrics[name])
    assert row["threshold"] == pytest.approx(0.63)
    assert captured["chart"] == {name: metrics[name] for name in METRIC_NAMES}
    assert captured["path"] == plot_path


def test_save_artifacts_confusion_sums_to_batch_size(tmp_path, monkeypatch):
    import pandas as pd

    monkeypatch.setattr(
        "model.train.plot_test_comparison",
        lambda chart, **kw: None,
    )

    logits = [5.0, -5.0, 5.0, 5.0, -5.0, -5.0, 5.0]
    labels = [1, 0, 0, 1, 1, 0, 1]
    result = run(logits, labels)
    metrics = compute_all(result["preds"], result["targets"])

    csv_path = tmp_path / "m.csv"
    save_artifacts(metrics, 0.5, csv_path=csv_path, plot_path=tmp_path / "m.png")

    row = pd.read_csv(csv_path).iloc[0]
    assert row["tp"] + row["fp"] + row["tn"] + row["fn"] == len(labels)