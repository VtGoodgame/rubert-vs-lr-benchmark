"""Тесты TextTokenizer.

Настоящий rubert не качается: AutoTokenizer.from_pretrained подменяется
заглушкой, поэтому тест проверяет пробрасывание параметров, а не словарь.
"""

import pytest
import torch

from model.tokenization import tokenizer as tokenizer_module

# Маркер снимает эти тесты в CI: они проверяют код моделей, а не независимые
# функции. Локально `pytest` без -m гоняет их на любом устройстве.
pytestmark = pytest.mark.torch


def test_init_stores_max_length(monkeypatch):
    monkeypatch.setattr(tokenizer_module.AutoTokenizer, "from_pretrained", lambda *a, **k: None)
    tok = tokenizer_module.TextTokenizer(model_name="fake/model", max_length=64)
    assert tok.max_length == 64


def test_init_defaults_to_128(monkeypatch):
    monkeypatch.setattr(tokenizer_module.AutoTokenizer, "from_pretrained", lambda *a, **k: None)
    tok = tokenizer_module.TextTokenizer(model_name="fake/model")
    assert tok.max_length == 128


def test_single_call_passes_truncation_and_padding(monkeypatch):
    seen = {}

    def fake_from_pretrained(*args, **kwargs):
        def call(text, **call_kwargs):
            seen.update(call_kwargs)
            return {"input_ids": torch.zeros((1, 4), dtype=torch.long)}

        class T:
            def __call__(self, text, **kw):
                return call(text, **kw)

        return T()

    monkeypatch.setattr(tokenizer_module.AutoTokenizer, "from_pretrained", fake_from_pretrained)
    tok = tokenizer_module.TextTokenizer(model_name="fake/model", max_length=32)
    tok("привет")

    assert seen["truncation"] is True
    assert seen["padding"] == "max_length"
    assert seen["max_length"] == 32


def test_batch_returns_batched_tensors(monkeypatch):
    class FakeHF:
        def __call__(self, texts, **kwargs):
            n = len(texts)
            return {
                "input_ids": torch.zeros((n, 5), dtype=torch.long),
                "attention_mask": torch.ones((n, 5), dtype=torch.long),
            }

    monkeypatch.setattr(
        tokenizer_module.AutoTokenizer, "from_pretrained", lambda *a, **k: FakeHF()
    )
    tok = tokenizer_module.TextTokenizer(model_name="fake/model", max_length=5)
    enc = tok.batch(["a", "b", "c"])

    assert enc["input_ids"].shape == (3, 5)
    assert enc["attention_mask"].shape == (3, 5)


def test_vocab_size_is_exposed(monkeypatch):
    class FakeHF:
        vocab_size = 30522

    monkeypatch.setattr(
        tokenizer_module.AutoTokenizer, "from_pretrained", lambda *a, **k: FakeHF()
    )
    tok = tokenizer_module.TextTokenizer(model_name="fake/model")
    assert tok.vocab_size == 30522


def test_no_network_call_on_init(monkeypatch):
    """from_pretrained обязан быть подменён — иначе тест ушёл бы в сеть."""
    def explode(*args, **kwargs):
        raise AssertionError("тест попытался скачать модель из сети")

    monkeypatch.setattr(tokenizer_module.AutoTokenizer, "from_pretrained", explode)
    with pytest.raises(AssertionError):
        tokenizer_module.TextTokenizer(model_name="DeepPavlov/rubert-base-cased")
