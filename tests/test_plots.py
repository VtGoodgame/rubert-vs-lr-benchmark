"""Тесты построения графиков.

Проверяем не красоту картинки, а факт: файл создан и не пустой.
"""

import matplotlib

matplotlib.use("Agg")

import pytest

from scripts.tracking.plots import plot_test_comparison, plot_training_curves

EPOCHS = [1, 2, 3]


def test_plot_training_curves_creates_file(tmp_path):
    out = tmp_path / "curves.png"
    result = plot_training_curves(
        EPOCHS,
        train_loss=[0.7, 0.5, 0.4],
        val_f1=[0.6, 0.75, 0.8],
        val_precision=[0.7, 0.8, 0.85],
        val_recall=[0.6, 0.7, 0.75],
        save_path=out,
    )
    assert result == str(out)
    assert out.exists()
    assert out.stat().st_size > 0


def test_plot_training_curves_without_loss(tmp_path):
    """Только метрики — без второй оси Y."""
    out = tmp_path / "metrics.png"
    plot_training_curves(EPOCHS, val_f1=[0.6, 0.75, 0.8], save_path=out)
    assert out.exists()


def test_plot_training_curves_creates_parent_dir(tmp_path):
    out = tmp_path / "nested" / "dir" / "curves.png"
    plot_training_curves(EPOCHS, train_loss=[0.7, 0.5, 0.4], save_path=out)
    assert out.exists()


def test_plot_test_comparison_creates_file(tmp_path):
    out = tmp_path / "compare.png"
    metrics = {"precision": 0.9, "recall": 0.8, "f1": 0.85, "accuracy": 0.95}
    result = plot_test_comparison(metrics, save_path=out)
    assert result == str(out)
    assert out.exists()
    assert out.stat().st_size > 0


def test_plot_test_comparison_rejects_empty():
    with pytest.raises(ValueError, match="пуст"):
        plot_test_comparison({})
