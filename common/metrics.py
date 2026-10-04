# common/metrics.py
"""Расчёт метрик бинарной классификации.

Модуль лежит в common/, а не в scripts/: его считают и baseline, и RuBERT, и
считать они должны одним и тем же кодом. Пока у каждой модели была своя копия,
метрики расходились на вырожденных данных — у baseline деление на ноль давало
0.0, а sklearn печатал предупреждение и тоже ставил 0.0, но по разным причинам.

Зависимостей нет ни одной, включая torch и pandas: поэтому его можно звать и из
model, и из scripts, не нарушая правило слоёв.
"""

from common.config import METRIC_NAMES

# Порядок колонок в CSV обеих моделей. Общий список нужен, чтобы файлы
# baseline_metrics.csv и rubert_metrics.csv читались одинаково и их можно было
# сравнивать глазами, а не только через compare.py.
CSV_COLUMNS = METRIC_NAMES + ("threshold", "tp", "fp", "tn", "fn")

# Сетка порогов для подбора рабочей точки: 0.05…0.95 с шагом 0.05.
# Крайние 0.0 и 1.0 не берём — они вырожденные (всё «spam» или всё «ham»).
DEFAULT_THRESHOLD_GRID = tuple(i / 20 for i in range(1, 20))


def compute_confusion(preds: list[int], targets: list[int]) -> dict:
    """
    Считает TP, FP, TN, FN для бинарной классификации.

    preds:   список предсказаний (0/1)
    targets: список истинных меток (0/1)

    strict=True: разная длина списков — ошибка, а не молчаливое усечение.
    Иначе потерянные примеры тихо испортили бы метрики.
    """
    pairs = list(zip(preds, targets, strict=True))
    tp = sum(1 for p, t in pairs if p == 1 and t == 1)
    fp = sum(1 for p, t in pairs if p == 1 and t == 0)
    tn = sum(1 for p, t in pairs if p == 0 and t == 0)
    fn = sum(1 for p, t in pairs if p == 0 and t == 1)
    return {"tp": tp, "fp": fp, "tn": tn, "fn": fn}


def compute_metrics(tp: int, fp: int, tn: int, fn: int) -> dict:
    """Считает precision, recall, f1, accuracy из confusion matrix."""
    precision = tp / (tp + fp) if (tp + fp) > 0 else 0.0
    recall = tp / (tp + fn) if (tp + fn) > 0 else 0.0
    f1 = 2 * precision * recall / (precision + recall) if (precision + recall) > 0 else 0.0
    accuracy = (tp + tn) / (tp + tn + fp + fn) if (tp + tn + fp + fn) > 0 else 0.0
    return {
        "precision": precision,
        "recall": recall,
        "f1": f1,
        "accuracy": accuracy,
    }


def compute_all(preds: list[int], targets: list[int]) -> dict:
    """Удобная обёртка: confusion + метрики одной функцией."""
    conf = compute_confusion(preds, targets)
    metrics = compute_metrics(**conf)
    return {**conf, **metrics}


def ordered_row(metrics: dict, threshold: float, extra: dict | None = None) -> dict:
    """Строка CSV в общем для обеих моделей порядке колонок.

    Модели пишут метрики в разные словари — у одной compute_at_threshold, у
    другой compute_all, — и без общего порядка колонки в baseline_metrics.csv и
    rubert_metrics.csv шли бы по-разному. Читать такие файлы рядом неудобно.
    Дополнительные параметры конкретной модели (у baseline это C) передаются
    в extra и встают в конец.
    """
    row = {name: float(metrics[name]) for name in METRIC_NAMES}
    row["threshold"] = float(threshold)
    for name in ("tp", "fp", "tn", "fn"):
        row[name] = int(metrics[name])
    if extra:
        row.update(extra)
    return row


def compute_at_threshold(
    probs: list[float],
    targets: list[int],
    threshold: float,
) -> dict:
    """Метрики при заданном пороге решения.

    Нужен именно список вероятностей, а не готовые метки: порог подбирается
    на val, поэтому метки на этом шаге ещё не решены.
    """
    preds = [int(p > threshold) for p in probs]
    return compute_all(preds, targets)


def find_best_threshold(
    probs: list[float],
    targets: list[int],
    grid: tuple[float, ...] = DEFAULT_THRESHOLD_GRID,
) -> tuple[float, dict]:
    """Перебирает пороги и возвращает лучший по f1 вместе с его метриками.

    Порог подбирается на val, а метрики потом считаются на test — иначе порог
    был бы выбран на тех же данных, на которых отчитываются, и f1 вышел бы
    завышенным.

    Сравнение строгое (>) и сетка идёт по возрастанию, поэтому при равном f1
    побеждает меньший порог: выбор воспроизводим от запуска к запуску.
    """
    if not probs:
        raise ValueError("список вероятностей пуст")
    if not grid:
        raise ValueError("сетка порогов пуста")

    best_threshold = grid[0]
    best_metrics = compute_at_threshold(probs, targets, best_threshold)

    for threshold in grid[1:]:
        metrics = compute_at_threshold(probs, targets, threshold)
        if metrics["f1"] > best_metrics["f1"]:
            best_threshold, best_metrics = threshold, metrics

    return best_threshold, best_metrics