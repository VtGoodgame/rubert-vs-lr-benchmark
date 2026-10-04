"""Тесты построения графиков.

Проверяем не красоту картинки, а факт: файл создан и не пустой.
"""

import matplotlib

matplotlib.use("Agg")

import matplotlib.pyplot as plt
import pytest

import common.plots as plots_module
from common.config import METRIC_NAMES
from common.plots import plot_model_comparison, plot_test_comparison, plot_training_curves

EPOCHS = [1, 2, 3]

METRICS = {"precision": 0.9, "recall": 0.8, "f1": 0.85, "accuracy": 0.95}


def capture_chart(monkeypatch, metrics, title):
    """Рисует график одной модели и возвращает его структуру.

    Файл закрывается до возврата, поэтому снимок делается в подменённом _save.
    Так проверяется сама диаграмма, а не факт наличия PNG.
    """
    snapshot = {}

    def fake_save(fig, path):
        ax = fig.axes[0]
        snapshot["bars"] = len(ax.patches)
        snapshot["xticks"] = [tick.get_text() for tick in ax.get_xticklabels()]
        snapshot["ylim"] = tuple(ax.get_ylim())
        snapshot["figsize"] = tuple(fig.get_size_inches())
        snapshot["heights"] = [round(patch.get_height(), 6) for patch in ax.patches]
        snapshot["legend"] = ax.get_legend() is not None
        plt.close(fig)
        return str(path)

    monkeypatch.setattr(plots_module, "_save", fake_save)
    plot_test_comparison(metrics, title=title, save_path="unused.png")
    return snapshot


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


def test_plot_model_comparison_draws_both_models(tmp_path):
    """Смысл функции — две модели рядом, поэтому обе должны попасть в график."""
    out = tmp_path / "comparison.png"
    results = {
        "TF-IDF + LR": dict(METRICS, f1=0.84),
        "RuBERT": dict(METRICS, f1=0.93),
    }
    result = plot_model_comparison(results, save_path=out)
    assert result == str(out)
    assert out.exists()
    assert out.stat().st_size > 0


def test_plot_model_comparison_accepts_single_model(tmp_path):
    out = tmp_path / "one.png"
    plot_model_comparison({"RuBERT": METRICS}, save_path=out)
    assert out.exists()


def test_plot_model_comparison_rejects_empty_results(tmp_path):
    with pytest.raises(ValueError, match="пуст"):
        plot_model_comparison({}, save_path=tmp_path / "x.png")


def test_plot_model_comparison_rejects_empty_metrics(tmp_path):
    with pytest.raises(ValueError, match="пуст"):
        plot_model_comparison({"RuBERT": METRICS}, metrics=(), save_path=tmp_path / "x.png")


def test_plot_model_comparison_reports_missing_metric(tmp_path):
    """Модель без f1 — это ошибка данных, а не молчаливое исключение метрики."""
    with pytest.raises(ValueError, match="f1"):
        plot_model_comparison(
            {"RuBERT": {"precision": 0.9, "recall": 0.8, "accuracy": 0.9}},
            save_path=tmp_path / "x.png",
        )


def test_plot_model_comparison_ignores_extra_keys(tmp_path):
    """В словаре метрик лежат ещё threshold и целые счётчики tp/fp/tn/fn —
    на графике они не должны появляться, даже если передать весь словарь."""
    noisy = dict(METRICS, threshold=0.63, tp=120, fp=4)
    out = tmp_path / "noisy.png"
    plot_model_comparison({"RuBERT": noisy}, metrics=["f1", "accuracy"], save_path=out)
    assert out.exists()


def test_plot_model_comparison_accepts_metric_subset(tmp_path):
    out = tmp_path / "subset.png"
    plot_model_comparison({"RuBERT": METRICS}, metrics=["f1"], save_path=out)
    assert out.exists()


def test_plot_test_comparison_hides_single_model_legend(tmp_path):
    """У одной модели легенда «модель» только занимает место."""
    out = tmp_path / "single.png"
    plot_test_comparison(METRICS, save_path=out, title="Одна модель")
    assert out.exists()


def test_per_model_charts_have_same_measurements(monkeypatch):
    """Графики baseline и RuBERT обязаны совпадать по длине.

    Раньше у RuBERT не было accuracy, и её график был на столбец короче. Внешне
    два графика одной и той же метрики выглядели по-разному, и сравнить их
    глазами было нельзя.
    """
    baseline = capture_chart(monkeypatch, METRICS, "baseline")
    rubert = capture_chart(monkeypatch, dict(METRICS, f1=0.93), "rubert")

    assert baseline["bars"] == len(METRIC_NAMES)
    assert baseline["bars"] == rubert["bars"]


def test_per_model_charts_are_visually_identical_except_values(monkeypatch):
    """Одинаковые подписи, шкала, размер и число столбцов — отличаются только
    высоты. Набор метрик задаётся METRIC_NAMES, а не тем, что попало в словарь."""
    baseline = capture_chart(monkeypatch, METRICS, "baseline")
    rubert = capture_chart(monkeypatch, dict(METRICS, precision=0.7), "rubert")

    for key in ("bars", "xticks", "ylim", "figsize", "legend"):
        assert baseline[key] == rubert[key], key
    assert baseline["xticks"] == list(METRIC_NAMES)
    assert baseline["legend"] is False


def test_per_model_chart_heights_are_the_metric_values(monkeypatch):
    """Столбец каждой метрики равен её значению, и порядок — METRIC_NAMES."""
    values = {"precision": 0.11, "recall": 0.22, "f1": 0.33, "accuracy": 0.44}
    snapshot = capture_chart(monkeypatch, values, "model")
    assert snapshot["heights"] == [values[name] for name in METRIC_NAMES]
