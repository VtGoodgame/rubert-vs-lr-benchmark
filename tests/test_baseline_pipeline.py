"""Тест пайплайна baseline: TF-IDF + Logistic Regression.

Проверяем на синтетике, что пайплайн не падает и метрики сходятся
с confusion matrix. Качество модели не проверяем — оно зависит от данных.
"""

import pandas as pd
import pytest
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression

from model.data.labels import encode_labels
from model.preprocessing.clean import clean_text
from scripts.tracking.metrics import compute_all

SPAM_WORDS = [
    "выигрыш миллион",
    "бесплатный приз",
    "срочно переведите деньги",
    "выиграли в лотерею",
    "быстрый кредит без проверки",
]
HAM_WORDS = [
    "встреча завтра в десять",
    "отправил отчет по проекту",
    "спасибо за помощь",
    "обед в кафе в полдень",
    "договор подписан",
]


def make_frame() -> pd.DataFrame:
    """Равные по размеру классы, чтобы метрики не искажались перекосом."""
    rows = [{"target": "spam", "text": f"{w} номер {i}"} for i, w in enumerate(SPAM_WORDS)]
    rows += [{"target": "ham", "text": f"{w} номер {i}"} for i, w in enumerate(HAM_WORDS)]
    return pd.DataFrame(rows)


def run_pipeline(frame):
    """TF-IDF + LR, как в scripts/baseline.py, и метрики на своих же данных."""
    X = [clean_text(str(t)) for t in frame["text"]]
    y = encode_labels(frame)

    vec = TfidfVectorizer(max_features=1000, ngram_range=(1, 1))
    X_vec = vec.fit_transform(X)

    clf = LogisticRegression(max_iter=200)
    clf.fit(X_vec, y)
    preds = clf.predict(X_vec)

    return compute_all(list(preds), list(y)), len(X)


def test_pipeline_learns_separable_words():
    metrics, n = run_pipeline(make_frame())
    assert metrics["f1"] == 1.0
    assert metrics["accuracy"] == 1.0
    assert metrics["tp"] + metrics["tn"] == n


def test_confusion_matrix_sums_to_sample_count():
    metrics, n = run_pipeline(make_frame())
    total = metrics["tp"] + metrics["tn"] + metrics["fp"] + metrics["fn"]
    assert total == n


def test_all_spam_detected_no_false_positives():
    metrics, _ = run_pipeline(make_frame())
    assert metrics["fn"] == 0
    assert metrics["fp"] == 0


def test_metrics_are_in_unit_range():
    metrics, _ = run_pipeline(make_frame())
    for name in ("precision", "recall", "f1", "accuracy"):
        assert 0.0 <= metrics[name] <= 1.0


def test_encoded_labels_are_balanced():
    frame = make_frame()
    y = encode_labels(frame)
    assert list(y) == [1] * 5 + [0] * 5


def test_single_class_raises_instead_of_guessing():
    """На одном классе LogisticRegression не обучается — падает, а не выдумывает метрики.

    Раньше считалось, что вернутся нули; фактически sklearn требует минимум
    два класса. Тест фиксирует реальное поведение, чтобы падение на
    одноклассовом срезе не выглядело загадочно.
    """
    frame = pd.DataFrame([{"target": "ham", "text": "только ham"} for _ in range(3)])
    with pytest.raises(ValueError, match="classes"):
        run_pipeline(frame)


def test_compute_all_handles_single_class_metrics():
    """А вот compute_all на пустой матрице нули отдаёт спокойно."""
    metrics = compute_all([0, 0], [1, 1])
    assert metrics["recall"] == 0.0
    assert metrics["f1"] == 0.0
    assert metrics["accuracy"] == 0.0
