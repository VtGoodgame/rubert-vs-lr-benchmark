"""Тесты SpamDataset: чтение CSV, кэш и выдача батчей.

Данные — во временном каталоге, токенизатор — заглушка из conftest.
Настоящий rubert не скачивается.
"""

import pandas as pd
import pytest
import torch

from model.data.dataset import SpamDataset

# Маркер снимает эти тесты в CI: они проверяют код моделей, а не независимые
# функции. Локально `pytest` без -m гоняет их на любом устройстве.
pytestmark = pytest.mark.torch


def test_len_matches_row_count(tmp_csv, spam_ham_rows, fake_tokenizer):
    ds = SpamDataset(str(tmp_csv(spam_ham_rows)), tokenizer=fake_tokenizer)
    assert len(ds) == len(spam_ham_rows)


def test_getitem_returns_expected_keys(tmp_csv, spam_ham_rows, fake_tokenizer):
    ds = SpamDataset(str(tmp_csv(spam_ham_rows)), tokenizer=fake_tokenizer)
    item = ds[0]
    assert set(item) == {"input_ids", "attention_mask", "labels"}


def test_labels_are_float32(tmp_csv, spam_ham_rows, fake_tokenizer):
    ds = SpamDataset(str(tmp_csv(spam_ham_rows)), tokenizer=fake_tokenizer)
    assert ds.labels.dtype == torch.float32


def test_labels_encode_ham_as_zero(tmp_csv, spam_ham_rows, fake_tokenizer):
    ds = SpamDataset(str(tmp_csv(spam_ham_rows)), tokenizer=fake_tokenizer)
    ham_index = next(i for i, r in enumerate(spam_ham_rows) if r["target"] == "ham")
    spam_index = next(i for i, r in enumerate(spam_ham_rows) if r["target"] == "spam")
    assert ds.labels[ham_index].item() == 0.0
    assert ds.labels[spam_index].item() == 1.0


def test_input_ids_padded_to_max_length(tmp_csv, spam_ham_rows, fake_tokenizer):
    ds = SpamDataset(str(tmp_csv(spam_ham_rows)), tokenizer=fake_tokenizer)
    assert ds.input_ids.shape[1] == fake_tokenizer.max_length


def test_nan_text_becomes_empty_string(tmp_csv, fake_tokenizer):
    rows = [
        {"target": "ham", "text": "нормальный текст"},
        {"target": "spam", "text": None},
    ]
    ds = SpamDataset(str(tmp_csv(rows)), tokenizer=fake_tokenizer)
    assert len(ds) == 2
    assert torch.isnan(ds.input_ids.float()).sum() == 0


def test_clean_text_applied_before_tokenizing(tmp_path, fake_tokenizer):
    """URL должен исчезнуть из текста до токенизации."""
    rows = [{"target": "spam", "text": "заходи http://example.com сюда"}]
    path = tmp_path / "url.csv"
    pd.DataFrame(rows).to_csv(path, index=False)

    seen_texts = []
    original_batch = fake_tokenizer.batch

    def spy(texts):
        seen_texts.extend(texts)
        return original_batch(texts)

    fake_tokenizer.batch = spy
    SpamDataset(str(path), tokenizer=fake_tokenizer)

    assert "[URL]" in seen_texts[0]
    assert "example.com" not in seen_texts[0]


def test_unknown_label_raises(tmp_csv, fake_tokenizer):
    rows = [{"target": "неизвестно", "text": "текст"}]
    with pytest.raises(ValueError):
        SpamDataset(str(tmp_csv(rows)), tokenizer=fake_tokenizer)


def test_cache_is_written(tmp_path, spam_ham_rows, fake_tokenizer):
    cache = tmp_path / "cache" / "train.pt"
    csv_path = tmp_path / "data.csv"
    pd.DataFrame(spam_ham_rows).to_csv(csv_path, index=False)

    SpamDataset(str(csv_path), tokenizer=fake_tokenizer, cache_path=str(cache))
    assert cache.exists()


def test_cache_reused_on_second_init(tmp_path, spam_ham_rows, fake_tokenizer):
    csv_path = tmp_path / "data.csv"
    pd.DataFrame(spam_ham_rows).to_csv(csv_path, index=False)
    cache = tmp_path / "train.pt"

    first = SpamDataset(str(csv_path), tokenizer=fake_tokenizer, cache_path=str(cache))
    calls_after_first = len(fake_tokenizer.calls)

    second = SpamDataset(str(csv_path), tokenizer=fake_tokenizer, cache_path=str(cache))

    assert len(fake_tokenizer.calls) == calls_after_first
    assert torch.equal(first.labels, second.labels)


def test_lazy_mode_tokenizes_per_item(tmp_csv, spam_ham_rows, fake_tokenizer):
    ds = SpamDataset(str(tmp_csv(spam_ham_rows)), tokenizer=fake_tokenizer, precompute=False)
    assert not hasattr(ds, "input_ids")
    item = ds[0]
    assert item["input_ids"].dim() == 1
