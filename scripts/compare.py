# scripts/compare.py
"""Сравнительный отчёт бенчмарка: RuBERT против TF-IDF + Logistic Regression.

Скрипт ничего не обучает — он читает артефакты, оставленные
scripts/baseline.py и model/train.py, и кладёт их рядом. Благодаря этому можно
сравнивать любые два прогона, а не только последние.

Протокол у моделей одинаковый: обучение на train_split, подбор порога на
val_split, отчёт на test.

Запуск (после обучения обеих моделей):
    python scripts/compare.py
"""

import os

import pandas as pd

from common.config import (
    BASELINE_METRICS_CSV,
    COMPARISON_CSV,
    COMPARISON_PLOT,
    METRIC_NAMES,
    RUBERT_METRICS_CSV,
    SPAM_THRESHOLD,
)
from common.logging_setup import get_logger, setup_logging
from common.plots import plot_model_comparison

logger = get_logger(__name__)

BASELINE_LABEL = "TF-IDF + Logistic Regression"
RUBERT_LABEL = "RuBERT-base-cased (fine-tune)"

# Откуда взять файл, если его нет
SOURCES = {
    BASELINE_METRICS_CSV: "python scripts/baseline.py",
    RUBERT_METRICS_CSV: "python -m model.train",
}


def load_metrics(csv_path: str | os.PathLike) -> dict[str, float]:
    """Читает CSV с метриками одной модели.

    Отдельная функция, а не чтение внутри main: так же файлы читает тест, без
    запуска скрипта целиком.
    """
    path = os.fspath(csv_path)
    if not os.path.exists(path):
        command = SOURCES.get(os.path.basename(path), "запустите обучение модели")
        raise FileNotFoundError(f"нет файла {path} — сначала {command}")

    row = pd.read_csv(path).iloc[0].to_dict()

    missing = [name for name in METRIC_NAMES if name not in row]
    if missing:
        raise ValueError(f"в {path} нет метрик: {', '.join(missing)}")

    return row


def build_comparison(
    baseline: dict[str, float],
    rubert: dict[str, float],
    metrics: tuple[str, ...] = METRIC_NAMES,
) -> pd.DataFrame:
    """Таблица «метрика | baseline | RuBERT | разница».

    Разница считается как rubert минус baseline: положительная означает, что
    RuBERT лучше.
    """
    return pd.DataFrame([
        {
            "metric": name,
            BASELINE_LABEL: float(baseline[name]),
            RUBERT_LABEL: float(rubert[name]),
            "delta": float(rubert[name]) - float(baseline[name]),
        }
        for name in metrics
    ])


def pick_winner(
    table: pd.DataFrame,
    thresholds: dict[str, float],
    metric: str = "f1",
) -> tuple[str, float]:
    """Модель с лучшим значением метрики и её собственный порог.

    Пороги передаются словарём, а не берутся из колонок таблицы: у каждой
    модели он свой, и в API должен попасть порог той модели, чьи метрики мы
    показываем как лучшие. Раньше пороги добавлялись в таблицу отдельными
    колонками, и их отсутствие приводило к KeyError вместо внятной ошибки.
    """
    row = table.loc[table["metric"] == metric].iloc[0]
    winner = RUBERT_LABEL if row[RUBERT_LABEL] > row[BASELINE_LABEL] else BASELINE_LABEL
    return winner, float(thresholds[winner])


def format_report(table: pd.DataFrame, winner: str, threshold: float) -> str:
    """Готовая таблица для консоли и подсказка, какое значение вписать в конфиг."""
    lines = [table.to_string(index=False, float_format=lambda v: f"{v:.4f}")]
    lines.append("")
    lines.append(f"Лучшая модель по f1: {winner}")
    lines.append(f"Её порог: {threshold:.2f}")
    lines.append(f"Впишите в common/config.py: SPAM_THRESHOLD = {threshold}")
    return "\n".join(lines)


def main() -> None:
    setup_logging()

    baseline = load_metrics(BASELINE_METRICS_CSV)
    rubert = load_metrics(RUBERT_METRICS_CSV)

    table = build_comparison(baseline, rubert)

    thresholds = {
        BASELINE_LABEL: float(baseline.get("threshold", SPAM_THRESHOLD)),
        RUBERT_LABEL: float(rubert.get("threshold", SPAM_THRESHOLD)),
    }
    table["baseline_threshold"] = thresholds[BASELINE_LABEL]
    table["rubert_threshold"] = thresholds[RUBERT_LABEL]

    table.to_csv(COMPARISON_CSV, index=False)

    # Один сравнительный график на обе модели: обе обучены по одному
    # протоколу на одних данных, поэтому сопоставление честное.
    plot_model_comparison(
        {
            BASELINE_LABEL: {name: float(baseline[name]) for name in METRIC_NAMES},
            RUBERT_LABEL: {name: float(rubert[name]) for name in METRIC_NAMES},
        },
        save_path=COMPARISON_PLOT,
    )

    winner, threshold = pick_winner(table, thresholds)
    report = format_report(table, winner, threshold)

    # scripts по умолчанию молчит на INFO, а отчёт без экрана бесполезен.
    print(report)
    logger.info("сравнение сохранено в %s и %s", COMPARISON_CSV, COMPARISON_PLOT)


if __name__ == "__main__":
    main()