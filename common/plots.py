"""Построение графиков метрик обучения.

Модуль ничего не знает про обучение и не импортирует torch — на вход
приходят уже посчитанные числа, на входе же они приходят из common.metrics.
Поэтому графики строятся и для baseline, и для RuBERT.

Живёт в common/, потому что график рисуют и scripts/, и model/: когда он лежал
в scripts/, model приходилось импортировать scripts, что запрещено правилом
слоёв. Общий модуль решает и это: обе модели рисуют свои графики одним и тем же
кодом, поэтому графики не могут разойтись ни в наборе метрик, ни во внешнем
виде — а раньше у RuBERT вовсе не было accuracy, и графики были разной длины.

Заголовки и подписи осей русские: графики читает человек, а не код.
"""

from __future__ import annotations

import os
from collections.abc import Sequence

import matplotlib.pyplot as plt
import numpy as np
import seaborn as sns

from common.config import METRIC_NAMES

# Метрики почти всегда в [0, 1], loss — нет. Поэтому у метрик своя ось.
LOSS_COLOR = "tab:blue"
METRIC_COLORS = {"val f1": "tab:green", "val precision": "tab:orange", "val recall": "tab:red"}


def _set_style() -> None:
    """Единый стиль всех графиков проекта."""
    sns.set_theme(style="whitegrid", context="talk", palette="deep")
    plt.rcParams.update(
        {
            "figure.figsize": (10, 6),
            "axes.titlesize": 16,
            "axes.labelsize": 14,
            "xtick.labelsize": 12,
            "ytick.labelsize": 12,
            "legend.fontsize": 12,
        }
    )


def _save(fig, save_path: str | os.PathLike[str]) -> str:
    """Сохраняет картинку и закрывает фигуру, иначе течёт память."""
    path = os.fspath(save_path)
    parent = os.path.dirname(path)
    if parent:
        os.makedirs(parent, exist_ok=True)
    fig.savefig(path, dpi=300, bbox_inches="tight")
    plt.close(fig)
    return path


def plot_training_curves(
    epochs: Sequence[int],
    train_loss: Sequence[float] | None = None,
    val_f1: Sequence[float] | None = None,
    val_precision: Sequence[float] | None = None,
    val_recall: Sequence[float] | None = None,
    title: str = "Кривые обучения",
    save_path: str | os.PathLike[str] = "training_curves.png",
) -> str:
    """Строит кривые обучения по эпохам.

    loss рисуется на левой оси, метрики — на правой: их масштабы не
    совпадают, и на одной оси f1 прижало бы loss к нулю.

    Путь к сохранённому файлу. Рисунок не показывается на экране —
    plt.show() в скриптах обучения мешал бы.
    """
    _set_style()
    fig, ax_loss = plt.subplots()
    n = len(epochs)

    has_loss = train_loss is not None and len(train_loss) == n
    if has_loss:
        ax_loss.plot(epochs, train_loss, label="train loss", color=LOSS_COLOR, linewidth=2)
        ax_loss.set_xlabel("Эпоха")
        ax_loss.set_ylabel("Loss", color=LOSS_COLOR)
        ax_loss.tick_params(axis="y", labelcolor=LOSS_COLOR)

    ax_metrics = ax_loss.twinx() if has_loss else ax_loss
    for name, values in (
        ("val f1", val_f1),
        ("val precision", val_precision),
        ("val recall", val_recall),
    ):
        if values is not None and len(values) == n:
            ax_metrics.plot(
                epochs,
                values,
                label=name,
                color=METRIC_COLORS[name],
                linewidth=2,
                linestyle="--",
            )

    if has_loss:
        ax_metrics.set_ylabel("Метрики")
        ax_metrics.set_ylim(0.0, 1.0)

    handles, legend_labels = ax_loss.get_legend_handles_labels()
    handles2, labels2 = ax_metrics.get_legend_handles_labels()
    ax_loss.legend(handles + handles2, legend_labels + labels2, loc="lower left")

    ax_loss.set_title(title)
    return _save(fig, save_path)


def plot_test_comparison(
    metrics: dict[str, float],
    title: str = "Метрики на test",
    save_path: str | os.PathLike[str] = "test_comparison.png",
) -> str:
    """Столбчатая диаграмма метрик одной модели.

    metrics: {"precision": 0.9, "recall": 0.8, "f1": 0.85}

    Путь к сохранённому файлу.
    """
    if not metrics:
        raise ValueError("metrics_dict пуст")

    # Сначала известные доли в принятом порядке, потом всё остальное — чтобы
    # графики разных моделей читались одинаково.
    names = [name for name in METRIC_NAMES if name in metrics] or list(metrics)
    single = {"модель": metrics}
    return plot_model_comparison(
        single,
        metrics=names,
        title=title,
        save_path=save_path,
        show_legend=False,
    )


def plot_model_comparison(
    results: dict[str, dict[str, float]],
    metrics: Sequence[str] = METRIC_NAMES,
    title: str = "Сравнение моделей на test",
    save_path: str | os.PathLike[str] = "comparison.png",
    show_legend: bool = True,
) -> str:
    """Сгруппированная столбчатая диаграмма: по метрике одна группа, в ней — модели.

    results: {"TF-IDF + LR": {"f1": 0.84, ...}, "RuBERT": {"f1": 0.91, ...}}

    Рисуем столбцы вручную, а не через sns.barplot: нужны подписи значений
    над каждым столбцом, а при hue их позиции зависят от seaborn.

    Путь к сохранённому файлу.
    """
    if not results:
        raise ValueError("results_dict пуст")
    if not metrics:
        raise ValueError("список метрик пуст")

    for model, model_metrics in results.items():
        missing = [name for name in metrics if name not in model_metrics]
        if missing:
            raise ValueError(f"у модели {model!r} нет метрик: {', '.join(missing)}")

    _set_style()
    fig, ax = plt.subplots()

    n_models = len(results)
    x = np.arange(len(metrics))
    width = 0.8 / n_models

    for index, (model, model_metrics) in enumerate(results.items()):
        values = [float(model_metrics[name]) for name in metrics]
        # Смещение так, чтобы группа столбцов стояла по центру своей метрики.
        offset = (index - (n_models - 1) / 2) * width
        bars = ax.bar(x + offset, values, width, label=model)
        ax.bar_label(bars, fmt="%.3f", padding=3, fontsize=11)

    ax.set_xticks(x)
    ax.set_xticklabels(metrics)
    ax.set_ylim(0.0, 1.0)
    ax.set_xlabel("")
    ax.set_ylabel("Значение")
    ax.set_title(title)

    if show_legend:
        ax.legend()

    return _save(fig, save_path)
