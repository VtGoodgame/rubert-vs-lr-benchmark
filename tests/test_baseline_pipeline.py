"""Тесты baseline: TF-IDF + Logistic Regression и выгрузка артефактов.

Тест зовёт настоящие функции scripts/baseline.py на синтетике: раньше пайплайн
был продублирован в тесте, поэтому проверялся не тот код, который запускается.
Качество модели не проверяется — оно зависит от данных.

Протокол как в бенчмарке: обучение на train, подбор порога на val, метрики на
test. Поэтому фикстура делает три файла, а не два.
"""

import matplotlib
import pandas as pd
import pytest

matplotlib.use("Agg")

from common.metrics import compute_all
from scripts import baseline as baseline_module
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


def _rows(words, filler, offset):
    """Письма одного класса: три сигнальных слова плюс общий filler.

    Три слова и общий filler нужны для правдоподобной уверенности модели. На
    одном слове на письмо TF-IDF даёт почти равные веса всем словам, вероятности
    сходятся к 0.5, и порог, выбранный на val, не переносится на test.

    Filler и набор слов общие для train, val и test — иначе модель видела бы в
    test незнакомые слова. Отличаются только индексы, поэтому письма в выборках
    не совпадают дословно.
    """
    rows = []
    for i in range(24):
        picked = " ".join(words[(offset + i + k) % len(words)] for k in range(3))
        rows.append({"text": f"{filler} {picked} {offset + i}"})
    return rows


@pytest.fixture
def splits(tmp_csv):
    """train, val и test на разделяемых словах, файлы во временном каталоге."""
    def make(offset):
        spam = [{"target": "spam", **r} for r in _rows(SPAM_WORDS, "письмо", offset)]
        ham = [{"target": "ham", **r} for r in _rows(HAM_WORDS, "письмо", offset)]
        return spam + ham

    return (
        tmp_csv(make(0), "train.csv"),
        tmp_csv(make(1000), "val.csv"),
        tmp_csv(make(2000), "test.csv"),
    )


def test_run_baseline_returns_metrics_and_predictions(splits):
    metrics, predictions = run_baseline(*splits, max_features=1000)

    assert set(metrics) >= {"precision", "recall", "f1", "accuracy", "threshold"}
    assert isinstance(predictions, pd.DataFrame)
    assert len(predictions) == 48   # по 24 письма каждого класса в test


def test_learns_separable_words(splits):
    metrics, _ = run_baseline(*splits, max_features=1000)
    assert metrics["f1"] == 1.0
    assert metrics["accuracy"] == 1.0


def test_confusion_matrix_sums_to_sample_count(splits):
    metrics, predictions = run_baseline(*splits, max_features=1000)
    total = metrics["tp"] + metrics["tn"] + metrics["fp"] + metrics["fn"]
    assert total == len(predictions)


def test_metrics_are_in_unit_range(splits):
    metrics, _ = run_baseline(*splits, max_features=1000)
    for name in ("precision", "recall", "f1", "accuracy"):
        assert 0.0 <= metrics[name] <= 1.0


def test_threshold_comes_from_val_not_from_test(splits):
    """Порог подбирается на val, поэтому в метриках он есть и лежит в сетке.

    Раньше порог был зашит в коде, и метрики на test считались при числе,
    выбранном без участия test.
    """
    metrics, _ = run_baseline(*splits, max_features=1000)
    assert metrics["threshold"] in [i / 20 for i in range(1, 20)]


def test_predictions_frame_has_expected_columns(splits):
    _, predictions = run_baseline(*splits, max_features=1000)
    assert list(predictions.columns) == ["target", "pred", "prob", "text"]


def test_predictions_frame_keeps_original_text(splits):
    _, predictions = run_baseline(*splits, max_features=1000)
    assert all(isinstance(text, str) and text for text in predictions["text"])


def test_prediction_columns_agree_with_threshold(splits):
    """Метка в predictions должна следовать из prob и порога, а не из чего-то ещё."""
    metrics, predictions = run_baseline(*splits, max_features=1000)
    expected = [
        "spam" if prob > metrics["threshold"] else "ham"
        for prob in predictions["prob"]
    ]
    assert predictions["pred"].tolist() == expected


def test_predictions_are_words_not_numbers(splits):
    """Baseline возвращает ответ почте: spam или ham, а не 0/1.

    Числа остаются в prob, а pred и target читаются человеком.
    """
    _, predictions = run_baseline(*splits, max_features=1000)
    assert set(predictions["pred"]) <= {"spam", "ham"}
    assert set(predictions["target"]) <= {"spam", "ham"}


def test_best_c_is_chosen_on_val(splits):
    """C отбирается на val — это аналог выбора эпохи у RuBERT.

    Метрики содержат выбранный C, и он лежит в сетке из конфига.
    """
    from common.config import C_GRID

    metrics, _ = run_baseline(*splits, max_features=1000)
    assert metrics["C"] in C_GRID


def test_select_c_on_val_is_deterministic_on_tie():
    """На разделяемых данных все C дают f1 = 1.0 на val.

    Правило то же, что и при выборе порога: строгое «>» при возрастании сетки,
    поэтому побеждает меньший C. Без этого выбор зависел бы от обхода сетки и
    менялся бы между запусками.
    """
    from sklearn.feature_extraction.text import TfidfVectorizer

    from scripts.baseline import select_c_on_val

    spam = ["выигрыш приз деньги"] * 5
    ham = ["встреча отчет проект"] * 5
    texts = spam + ham
    y = [1] * 5 + [0] * 5

    vec = TfidfVectorizer(ngram_range=(1, 2))
    x = vec.fit_transform(texts)

    c, model = select_c_on_val(x, y, x, y, grid=(0.01, 100.0))
    assert c == 0.01
    assert model.C == 0.01
    # Возвращается обученная модель, а не только выбранный параметр.
    assert model.predict(x).tolist() == y


def test_select_c_on_val_returns_model_fitted_on_train():
    """Отбор обязан вернуть уже обученную модель, а не только параметр:
    иначе run_baseline учил бы её второй раз на train."""
    import numpy as np

    from scripts.baseline import build_model, select_c_on_val

    x = np.array([[0.1, 0.9], [0.2, 0.8], [0.9, 0.1], [0.8, 0.2]])
    y = [1, 1, 0, 0]
    c, model = select_c_on_val(x, y, x, y, grid=(1.0,))
    assert isinstance(model, type(build_model()))
    assert c == 1.0


def test_load_splits_fills_missing_text(tmp_csv, spam_ham_rows):
    """Пропуски в text заменяются пустой строкой: TfidfVectorizer не ест None."""
    path = tmp_csv(spam_ham_rows, "with_gaps.csv")
    frame = pd.read_csv(path)
    frame.loc[0, "text"] = None
    frame.to_csv(path, index=False)

    train, val, test = load_splits(path, path, path)
    assert train["text"].isna().sum() == 0
    assert len(val) == len(spam_ham_rows)
    assert len(test) == len(spam_ham_rows)


def test_single_class_raises_instead_of_guessing(tmp_csv):
    """sklearn требует минимум два класса: падение внятнее выдуманных метрик."""
    rows = [{"target": "ham", "text": "только ham"} for _ in range(3)]
    path = tmp_csv(rows, "single.csv")
    with pytest.raises(ValueError, match="classes"):
        run_baseline(path, path, path)


def test_build_model_uses_benchmark_settings():
    """class_weight корректнее при перекосе классов, max_iter — для сходимости."""
    model = build_model()
    assert model.max_iter == 1000
    assert model.class_weight == "balanced"


def test_save_artifacts_writes_csv_and_png(splits, tmp_path):
    metrics, predictions = run_baseline(*splits, max_features=1000)

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
    metrics, predictions = run_baseline(*splits, max_features=1000)
    csv_path = tmp_path / "metrics.csv"
    save_artifacts(metrics, predictions, csv_path=csv_path,
                   predictions_path=tmp_path / "p.csv", plot_path=tmp_path / "m.png")

    saved = pd.read_csv(csv_path)
    assert saved.loc[0, "f1"] == pytest.approx(metrics["f1"])
    assert saved.loc[0, "threshold"] == pytest.approx(metrics["threshold"])


def test_save_artifacts_excludes_counts_and_threshold_from_chart(splits, tmp_path, monkeypatch):
    """В график идут только доли: целые tp/fn сломали бы шкалу [0, 1], а
    threshold — не метрика, а параметр, по которому эти доли посчитаны."""
    metrics, predictions = run_baseline(*splits, max_features=1000)

    captured = {}

    def capture(chart, **kwargs):
        captured["chart"] = chart
        return str(kwargs.get("save_path", ""))

    monkeypatch.setattr(baseline_module, "plot_test_comparison", capture)
    save_artifacts(metrics, predictions, csv_path=tmp_path / "m.csv",
                   predictions_path=tmp_path / "p.csv", plot_path=tmp_path / "m.png")

    assert set(captured["chart"]) == {"precision", "recall", "f1", "accuracy"}
    assert "threshold" not in captured["chart"]
    assert all(isinstance(value, float) for value in captured["chart"].values())


def test_metrics_contain_integer_counts():
    """Сами метрики включают целые tp/fp/tn/fn — их-то в график и не пускаем."""
    counts = compute_all([1, 0, 1], [1, 1, 0])
    assert counts["tp"] == 1
    assert counts["fp"] == 1
    assert isinstance(counts["tp"], int)