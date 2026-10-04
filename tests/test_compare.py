"""Тесты сравнительного отчёта scripts/compare.py.

Скрипт ничего не обучает, поэтому проверять тут нечему: он читает два CSV и
складывает их в таблицу и график. Качество моделей не проверяется — числа
подставляются вручную.
"""

import matplotlib
import pandas as pd
import pytest

matplotlib.use("Agg")

from common.config import METRIC_NAMES
from scripts import compare as compare_module
from scripts.compare import build_comparison, load_metrics, pick_winner


def _row(f1, precision=0.9, recall=0.8, accuracy=0.95, threshold=0.5):
    row = {"precision": precision, "recall": recall, "f1": f1, "accuracy": accuracy}
    row["threshold"] = threshold
    row.update({"tp": 100, "fp": 5, "tn": 200, "fn": 7})
    return row


def _write(tmp_path, name, row):
    path = tmp_path / name
    pd.DataFrame([row]).to_csv(path, index=False)
    return path


def test_load_metrics_reads_single_row(tmp_path):
    path = _write(tmp_path, "m.csv", _row(0.84))
    loaded = load_metrics(path)
    assert loaded["f1"] == pytest.approx(0.84)
    assert loaded["threshold"] == pytest.approx(0.5)


def test_load_metrics_points_at_missing_command(tmp_path):
    """Ошибка должна называть команду, а не только путь: иначе непонятно,
    что делать дальше."""
    path = tmp_path / compare_module.BASELINE_METRICS_CSV
    with pytest.raises(FileNotFoundError, match="baseline"):
        load_metrics(path)


def test_load_metrics_reports_missing_metric(tmp_path):
    """CSV без f1 — это неполный артефакт, и молча рисовать его нельзя."""
    path = tmp_path / "m.csv"
    pd.DataFrame([{"precision": 0.9}]).to_csv(path, index=False)
    with pytest.raises(ValueError, match="f1"):
        load_metrics(path)


def test_build_comparison_lists_every_metric():
    table = build_comparison(_row(0.84), _row(0.93))
    assert list(table["metric"]) == list(METRIC_NAMES)


def test_build_comparison_delta_is_rubert_minus_baseline():
    table = build_comparison(_row(0.80), _row(0.90)).set_index("metric")
    assert table.loc["f1", "delta"] == pytest.approx(0.10)


def test_build_comparison_delta_is_negative_when_baseline_wins():
    table = build_comparison(_row(0.90), _row(0.80)).set_index("metric")
    assert table.loc["f1", "delta"] == pytest.approx(-0.10)


def test_pick_winner_returns_rubert_threshold():
    """В API должен попасть порог той модели, чьи метрики мы назвали лучшими."""
    table = build_comparison(_row(0.80), _row(0.90))
    thresholds = {compare_module.BASELINE_LABEL: 0.45, compare_module.RUBERT_LABEL: 0.63}

    winner, threshold = pick_winner(table, thresholds)
    assert winner == compare_module.RUBERT_LABEL
    assert threshold == pytest.approx(0.63)


def test_pick_winner_returns_baseline_threshold():
    table = build_comparison(_row(0.95), _row(0.90))
    thresholds = {compare_module.BASELINE_LABEL: 0.30, compare_module.RUBERT_LABEL: 0.70}

    winner, threshold = pick_winner(table, thresholds)
    assert winner == compare_module.BASELINE_LABEL
    assert threshold == pytest.approx(0.30)


def test_pick_winner_tie_goes_to_baseline():
    """При равном f1 линейная модель проще, и победа отдаётся ей.

    Правило то же, что и при выборе порога: при равенстве выигрывает
    детерминированный вариант, а не тот, что обошёлся дороже.
    """
    table = build_comparison(_row(0.90), _row(0.90))
    thresholds = {compare_module.BASELINE_LABEL: 0.4, compare_module.RUBERT_LABEL: 0.6}

    winner, threshold = pick_winner(table, thresholds)
    assert winner == compare_module.BASELINE_LABEL
    assert threshold == pytest.approx(0.4)


@pytest.fixture
def artifacts(tmp_path, monkeypatch):
    """Два артефакта в tmp_path и подменённые на них пути скрипта."""
    baseline = _write(tmp_path, compare_module.BASELINE_METRICS_CSV, _row(0.84, threshold=0.45))
    rubert = _write(tmp_path, compare_module.RUBERT_METRICS_CSV, _row(0.93, threshold=0.63))

    csv_out = tmp_path / "comparison.csv"
    png_out = tmp_path / "comparison.png"
    monkeypatch.setattr(compare_module, "BASELINE_METRICS_CSV", str(baseline))
    monkeypatch.setattr(compare_module, "RUBERT_METRICS_CSV", str(rubert))
    monkeypatch.setattr(compare_module, "COMPARISON_CSV", str(csv_out))
    monkeypatch.setattr(compare_module, "COMPARISON_PLOT", str(png_out))
    return csv_out, png_out


def test_main_writes_table_and_plot(artifacts):
    """Сквозной сценарий: два артефакта на диске → таблица и график."""
    csv_out, png_out = artifacts

    compare_module.main()

    assert csv_out.exists()
    assert png_out.exists()
    assert png_out.stat().st_size > 0

    saved = pd.read_csv(csv_out)
    assert list(saved["metric"]) == list(METRIC_NAMES)
    assert {"baseline_threshold", "rubert_threshold"} <= set(saved.columns)


def test_main_prints_port_for_config(artifacts, capsys):
    """Отчёт без подсказки, какое значение вписать в конфиг, неполон."""
    compare_module.main()

    out = capsys.readouterr().out
    assert "SPAM_THRESHOLD = 0.63" in out
    assert compare_module.RUBERT_LABEL in out