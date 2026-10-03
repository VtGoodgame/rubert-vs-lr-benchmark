"""Тесты baseline: TF-IDF + Logistic Regression и выгрузка артефактов.

Тест зовёт настоящие функции scripts/baseline.py на синтетике: раньше пайплайн
был продублирован в тесте, поэтому проверялся не тот код, который запускается.
Качество модели не проверяется — оно зависит от данных.
"""

import matplotlib
import pandas as pd
import pytest

matplotlib.use("Agg")

from scripts.baseline import build_model, load_splits, run_baseline, save_artifacts

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


@pytest.fixture
def splits(tmp_csv):
    """train и test на разделяемых словах, файлы во временном каталоге."""
    train_rows = [{"target": "spam", "text": f"{w} номер {i}"} for i, w in enumerate(SPAM_WORDS)]
    train_rows += [{"target": "ham", "text": f"{w} номер {i}"} for i, w in enumerate(HAM_WORDS)]
    test_rows = [{"target": "spam", "text": f"{w} тест {i}"} for i, w in enumerate(SPAM_WORDS)]
    test_rows += [{"target": "ham", "text": f"{w} тест {i}"} for i, w in enumerate(HAM_WORDS)]
    return tmp_csv(train_rows, "train.csv"), tmp_csv(test_rows, "test.csv")


def test_run_baseline_returns_metrics_and_predictions(splits):
    train_path, test_path = splits
    metrics, predictions = run_baseline(train_path, test_path, max_features=1000)

    assert set(metrics) >= {"precision", "recall", "f1", "accuracy"}
    assert isinstance(predictions, pd.DataFrame)
    assert len(predictions) == 10


def test_learns_separable_words(splits):
    train_path, test_path = splits
    metrics, _ = run_baseline(train_path, test_path, max_features=1000)
    assert metrics["f1"] == 1.0
    assert metrics["accuracy"] == 1.0


def test_confusion_matrix_sums_to_sample_count(splits):
    train_path, test_path = splits
    metrics, predictions = run_baseline(train_path, test_path, max_features=1000)
    total = metrics["tp"] + metrics["tn"] + metrics["fp"] + metrics["fn"]
    assert total == len(predictions)


def test_metrics_are_in_unit_range(splits):
    train_path, test_path = splits
    metrics, _ = run_baseline(train_path, test_path, max_features=1000)
    for name in ("precision", "recall", "f1", "accuracy"):
        assert 0.0 <= metrics[name] <= 1.0


def test_predictions_frame_has_expected_columns(splits):
    train_path, test_path = splits
    _, predictions = run_baseline(train_path, test_path, max_features=1000)
    assert list(predictions.columns) == ["target", "pred", "text"]


def test_predictions_frame_keeps_original_text(splits):
    train_path, test_path = splits
    _, predictions = run_baseline(train_path, test_path, max_features=1000)
    assert all(isinstance(text, str) and text for text in predictions["text"])


def test_load_splits_fills_missing_text(tmp_csv, spam_ham_rows):
    """Пропуски в text заменяются пустой строкой: TfidfVectorizer не ест None."""
    path = tmp_csv(spam_ham_rows, "with_gaps.csv")
    frame = pd.read_csv(path)
    frame.loc[0, "text"] = None
    frame.to_csv(path, index=False)

    train, test = load_splits(path, path)
    assert train["text"].isna().sum() == 0
    assert len(test) == len(spam_ham_rows)


def test_single_class_raises_instead_of_guessing(tmp_csv):
    """sklearn требует минимум два класса: падение внятнее выдуманных метрик."""
    rows = [{"target": "ham", "text": "только ham"} for _ in range(3)]
    path = tmp_csv(rows, "single.csv")
    with pytest.raises(ValueError, match="classes"):
        run_baseline(path, path)


def test_build_model_uses_benchmark_settings():
    """class_weight корректнее при перекосе классов, max_iter — для сходимости."""
    model = build_model()
    assert model.max_iter == 1000
    assert model.class_weight == "balanced"


def test_save_artifacts_writes_csv_and_png(splits, tmp_path):
    train_path, test_path = splits
    metrics, predictions = run_baseline(train_path, test_path, max_features=1000)

    paths = save_artifacts(
        metrics,
        predictions,
        csv_path=tmp_path / "metrics.csv",
        predictions_path=tmp_path / "preds.csv",
        plot_path=tmp_path / "metrics.png",
    )

    assert (tmp_path / "metrics.csv").exists()
    assert (tmp_path / "preds.csv").exists()
    assert (tmp_path / "metrics.png").exists()
    assert (tmp_path / "metrics.png").stat().st_size > 0
    assert set(paths) == {"metrics_csv", "predictions_csv", "plot"}


def test_saved_metrics_csv_round_trips(splits, tmp_path):
    train_path, test_path = splits
    metrics, predictions = run_baseline(train_path, test_path, max_features=1000)
    csv_path = tmp_path / "metrics.csv"
    save_artifacts(metrics, predictions, csv_path=csv_path,
                   predictions_path=tmp_path / "p.csv", plot_path=tmp_path / "m.png")

    saved = pd.read_csv(csv_path)
    assert saved.loc[0, "f1"] == pytest.approx(metrics["f1"])


def test_save_artifacts_ignores_integer_counts_in_chart(splits, tmp_path):
    """В график идут только доли: tp/fn на шкале [0, 1] сломали бы разметку."""
    train_path, test_path = splits
    metrics, predictions = run_baseline(train_path, test_path, max_features=1000)
    save_artifacts(metrics, predictions, csv_path=tmp_path / "m.csv",
                   predictions_path=tmp_path / "p.csv", plot_path=tmp_path / "m.png")

    from scripts.tracking.metrics import compute_all

    chart_names = {"precision", "recall", "f1", "accuracy"}
    assert chart_names <= set(metrics)
    assert "tp" not in chart_names
    assert compute_all([1, 0], [1, 0])["tp"] == 1