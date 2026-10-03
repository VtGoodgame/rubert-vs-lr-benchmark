# scripts/baseline.py
"""Baseline: TF-IDF + Logistic Regression на тех же данных, что и RuBERT.

Считается f1, метрики сохраняются в CSV, график метрик кладётся в PNG.
Руками в коде ничего не печатается — только через logger.

Запуск:
    python scripts/baseline.py
"""

import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression

from common.logging_setup import get_logger, setup_logging
from model.data.labels import encode_labels
from model.preprocessing.clean import clean_text
from scripts.tracking.metrics import compute_all
from scripts.tracking.plots import plot_test_comparison

logger = get_logger(__name__)

TRAIN_PATH = "model/data/train_split.csv"
TEST_PATH = "model/data/test.csv"
MAX_FEATURES = 10000


def load_splits(train_path=TRAIN_PATH, test_path=TEST_PATH) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Читает train и test. Пустые тексты заменяются пустой строкой, иначе
    TfidfVectorizer падает на None."""
    train = pd.read_csv(train_path).fillna("")
    test = pd.read_csv(test_path).fillna("")
    return train, test


def build_model(max_features: int = MAX_FEATURES) -> LogisticRegression:
    """Модель с настройками бенчмарка: class_weight корректнее при перекосе,
    max_iter=1000 — иначе LR не сходится на биграммах."""
    return LogisticRegression(max_iter=1000, class_weight="balanced")


def run_baseline(
    train_path=TRAIN_PATH,
    test_path=TEST_PATH,
    max_features: int = MAX_FEATURES,
) -> tuple[pd.DataFrame, dict]:
    """Обучает TF-IDF+LR и возвращает метрики на test.

    Возвращает (метрики, предсказания). Метрики — словарь из compute_all,
    предсказания нужны для сохранения в CSV и разбора ошибок.
    """
    train, test = load_splits(train_path, test_path)

    X_train = [clean_text(str(text)) for text in train["text"]]
    y_train = encode_labels(train)
    X_test = [clean_text(str(text)) for text in test["text"]]
    y_test = encode_labels(test)

    logger.info("TF-IDF: признаков в словаре %d, примеров train %d", max_features, len(X_train))

    vectorizer = TfidfVectorizer(max_features=max_features, ngram_range=(1, 2))
    X_train_vec = vectorizer.fit_transform(X_train)
    X_test_vec = vectorizer.transform(X_test)

    model = build_model()
    model.fit(X_train_vec, y_train)
    preds = model.predict(X_test_vec)

    metrics = compute_all([int(p) for p in preds], [int(t) for t in y_test])
    logger.info("TF-IDF + LR F1: %.4f", metrics["f1"])

    predictions = pd.DataFrame({
        "target": y_test,
        "pred": preds,
        "text": test["text"].tolist(),
    })
    return metrics, predictions


def save_artifacts(
    metrics: dict,
    predictions: pd.DataFrame,
    csv_path="baseline_metrics.csv",
    predictions_path="baseline_predictions.csv",
    plot_path="baseline_metrics.png",
) -> dict[str, str]:
    """Сохраняет метрики в CSV, предсказания в CSV и график в PNG."""
    pd.DataFrame([metrics]).to_csv(csv_path, index=False)
    predictions.to_csv(predictions_path, index=False)
    chart = {name: float(value) for name, value in metrics.items() if isinstance(value, float)}
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