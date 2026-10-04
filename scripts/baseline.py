# scripts/baseline.py
"""Baseline: TF-IDF + Logistic Regression на тех же данных, что и RuBERT.

Протокол у двух моделей один, иначе сравнение бессмысленно:

    обучение   — train_split;
    отбор      — по f1 на val_split (у RuBERT это выбор эпохи, у LR — выбор C);
    порог      — по f1 на val_split, у каждой модели свой;
    отчёт      — test, единственное касание, на нём ничего не подбирается.

Отбор по val у обеих моделей — обязательная часть равенства. У RuBERT val
выбирает эпоху, у линейной модели раньше не выбиралось ничего: она просто
обучалась один раз. Теперь у неё перебирается сила регуляризации C, и
победитель выбирается тем же правилом, что и эпоха.

Предсказание возвращаются словами spam/ham, а не 0/1: Baseline — это модель для
почты, и её вывод должен читаться как ответ, а не как код. Числа остаются в
колонке prob, а метрики считаются внутри, где 0/1 удобнее.

Запуск:
    python scripts/baseline.py
"""

import os

import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression

from common.config import (
    BASELINE_METRICS_CSV,
    BASELINE_PLOT,
    BASELINE_PREDICTIONS_CSV,
    C_GRID,
    MAX_FEATURES,
    METRIC_NAMES,
    TEST_CSV,
    TRAIN_CSV,
    VAL_CSV,
)
from common.logging_setup import get_logger, setup_logging
from common.metrics import compute_at_threshold, find_best_threshold, ordered_row
from common.plots import plot_test_comparison
from model.data.labels import decode_labels, encode_labels
from model.preprocessing.clean import clean_text

logger = get_logger(__name__)


def load_splits(
    train_path: str | os.PathLike = TRAIN_CSV,
    val_path: str | os.PathLike = VAL_CSV,
    test_path: str | os.PathLike = TEST_CSV,
) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    """Читает train, val и test. Пустые тексты заменяются пустой строкой, иначе
    TfidfVectorizer падает на None."""
    train = pd.read_csv(train_path).fillna("")
    val = pd.read_csv(val_path).fillna("")
    test = pd.read_csv(test_path).fillna("")
    return train, val, test


def build_model(c: float = 1.0) -> LogisticRegression:
    """Модель с настройками бенчмарка.

    class_weight корректнее при перекосе классов, max_iter=1000 — иначе LR
    не сходится на биграммах. Размер словаря задаёт TfidfVectorizer, а не
    сама модель, поэтому параметра max_features здесь нет.
    """
    return LogisticRegression(max_iter=1000, class_weight="balanced", C=c)


def select_c_on_val(
    X_train_vec,
    y_train,
    X_val_vec,
    y_val,
    grid: tuple[float, ...] = C_GRID,
) -> tuple[float, LogisticRegression]:
    """Перебирает C и оставляет модель с лучшим f1 на val.

    Так же устроен выбор эпохи у RuBERT: кандидаты обучаются на train,
    отбор идёт по val. Поэтому у моделей одинаковый протокол отбора, а
    не только одинаковые данные.

    Сравнение строгое (>) при возрастании сетки, поэтому при равном f1
    побеждает меньший C — выбор воспроизводим от запуска к запуску.
    """
    best_c = grid[0]
    best_model = build_model(best_c)
    best_f1 = -1.0

    for c in grid:
        model = build_model(c)
        model.fit(X_train_vec, y_train)
        val_probs = [float(p) for p in model.predict_proba(X_val_vec)[:, 1]]
        _, val_metrics = find_best_threshold(val_probs, [int(t) for t in y_val])
        logger.info("C=%s val_f1=%.4f", c, val_metrics["f1"])
        if val_metrics["f1"] > best_f1:
            best_f1, best_c, best_model = val_metrics["f1"], c, model

    logger.info("выбран C=%s (val f1=%.4f)", best_c, best_f1)
    return best_c, best_model


def predict(model: LogisticRegression, X_vec, threshold: float) -> list[str]:
    """Предсказания словами: 'spam' или 'ham'.

    Порог явный, а не 0.5: он подобран на val, и в test-метриках использовался
    именно он. Неявный 0.5 здесь означал бы, что измеренное и предсказанное
    расходятся.
    """
    probs = model.predict_proba(X_vec)[:, 1]
    return decode_labels([int(p > threshold) for p in probs])


def run_baseline(
    train_path: str | os.PathLike = TRAIN_CSV,
    val_path: str | os.PathLike = VAL_CSV,
    test_path: str | os.PathLike = TEST_CSV,
    max_features: int = MAX_FEATURES,
) -> tuple[dict, pd.DataFrame]:
    """Обучает TF-IDF+LR и возвращает метрики на test.

    Порог подбирается на val, метрики считаются на test при нём. В словарь
    метрик кладётся и сам threshold — по сохранённому CSV видно, при каком
    пороге получены цифры.

    Возвращает (метрики, предсказания). Метрики — словарь из compute_at_threshold
    плюс threshold, предсказания нужны для сохранения в CSV и разбора ошибок.
    """
    train, val, test = load_splits(train_path, val_path, test_path)

    X_train = [clean_text(str(text)) for text in train["text"]]
    y_train = encode_labels(train)
    X_val = [clean_text(str(text)) for text in val["text"]]
    y_val = encode_labels(val)
    X_test = [clean_text(str(text)) for text in test["text"]]
    y_test = encode_labels(test)

    logger.info("TF-IDF: признаков в словаре %d, примеров train %d", max_features, len(X_train))

    vectorizer = TfidfVectorizer(max_features=max_features, ngram_range=(1, 2))
    X_train_vec = vectorizer.fit_transform(X_train)
    X_val_vec = vectorizer.transform(X_val)
    X_test_vec = vectorizer.transform(X_test)

    # Отбор C по val — аналог выбора эпохи у RuBERT.
    best_c, model = select_c_on_val(X_train_vec, y_train, X_val_vec, y_val)

    # predict_proba вместо predict: порог подбирается на val, а метки там
    # ещё не решены. Столбец 1 — вероятность spam.
    val_probs = [float(p) for p in model.predict_proba(X_val_vec)[:, 1]]
    test_probs = [float(p) for p in model.predict_proba(X_test_vec)[:, 1]]

    threshold, _ = find_best_threshold(val_probs, [int(t) for t in y_val])
    logger.info("порог по f1 на val: %.2f", threshold)

    metrics = compute_at_threshold(test_probs, [int(t) for t in y_test], threshold)
    metrics["threshold"] = threshold
    metrics["C"] = best_c
    logger.info("TF-IDF + LR F1 на test: %.4f", metrics["f1"])

    predictions = pd.DataFrame({
        "target": decode_labels(y_test),
        "pred": predict(model, X_test_vec, threshold),
        "prob": test_probs,
        "text": test["text"].tolist(),
    })
    return metrics, predictions


def save_artifacts(
    metrics: dict,
    predictions: pd.DataFrame,
    csv_path: str | os.PathLike = BASELINE_METRICS_CSV,
    predictions_path: str | os.PathLike = BASELINE_PREDICTIONS_CSV,
    plot_path: str | os.PathLike = BASELINE_PLOT,
) -> dict[str, str]:
    """Сохраняет метрики в CSV, предсказания в CSV и график в PNG."""
    row = ordered_row(metrics, metrics["threshold"], extra={"C": metrics["C"]})
    pd.DataFrame([row]).to_csv(csv_path, index=False)
    predictions.to_csv(predictions_path, index=False)

    # Доли отбираем по именам, а не по типу: threshold тоже float, и проверка
    # isinstance(value, float) затащила бы его в график.
    chart = {name: float(metrics[name]) for name in METRIC_NAMES}
    plot_test_comparison(chart, title="TF-IDF + Logistic Regression", save_path=plot_path)
    logger.info("метрики сохранены в %s и %s, график — %s", csv_path, predictions_path, plot_path)
    return {
        "metrics_csv": str(csv_path),
        "predictions_csv": str(predictions_path),
        "plot": str(plot_path),
    }


def main() -> None:
    setup_logging()
    metrics, predictions = run_baseline()
    save_artifacts(metrics, predictions)


if __name__ == "__main__":
    main()