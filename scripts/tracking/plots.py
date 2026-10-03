"""Построение графиков метрик обучения.

Модуль ничего не знает про обучение и не импортирует torch — на вход
приходят уже посчитанные числа, на выходе PNG-файлы. Поэтому графики
можно строить и для baseline, и для RuBERT.

Заголовки и подписи осей русские: графики читает человек, а не код.
"""

from __future__ import annotations

import os
from collections.abc import Sequence

import matplotlib.pyplot as plt
import seaborn as sns

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


def _save(fig, save_path: str | os.PathLike) -> str:
    """Сохраняет картинку и закрывает фигуру, иначе течёт память."""
    path = os.fspath(save_path)
    parent = os.path.dirname(path)
    os.makedirs(parent, exist_ok=True) if parent else None
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
    save_path: str | os.PathLike = "training_curves.png",
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
    save_path: str | os.PathLike = "test_comparison.png",
) -> str:
    """Строит столбчатую диаграмму метрик — например для сравнения моделей.

    metrics: {"precision": 0.9, "recall": 0.8, "f1": 0.85}

    Путь к сохранённому файлу.
    """
    if not metrics:
        raise ValueError("metrics_dict пуст")

    _set_style()
    names = list(metrics)
    values = [float(metrics[name]) for name in names]

    fig, ax = plt.subplots()
    sns.barplot(x=names, y=values, ax=ax, color=sns.color_palette("deep")[0])
    ax.set_ylim(0.0, 1.0)
    ax.set_xlabel("")
    ax.set_ylabel("Значение")
    ax.set_title(title)

    for x, value in enumerate(values):
        ax.text(x, value + 0.02, f"{value:.3f}", ha="center", va="bottom")

    return _save(fig, save_path)
