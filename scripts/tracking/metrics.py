# scripts/tracking/metrics.py
"""Расчёт метрик бинарной классификации. Без зависимостей от torch/pandas."""


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