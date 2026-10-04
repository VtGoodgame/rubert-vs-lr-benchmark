"""Тесты SpamClassifier: форма выхода и обучаемость.

AutoModel.from_pretrained подменяется модулем с маленьким энкодером —
настоящие веса rubert (700 МБ) в CI не скачиваются.
"""

import pytest
import torch
import torch.nn as nn
from transformers import AutoConfig

from model import spam_classifier as classifier_module
from model.spam_classifier import SpamClassifier

# Маркер снимает эти тесты в CI: они проверяют код моделей, а не независимые
# функции. Локально `pytest` без -m гоняет их на любом устройстве.
pytestmark = pytest.mark.torch

BATCH = 4
SEQ_LEN = 6
HIDDEN = 16


class FakeConfig:
    hidden_size = HIDDEN


class FakeEncoder(nn.Module):
    """Энкодер нужной формы: [batch, seq_len, hidden]."""

    config = FakeConfig()

    def __init__(self):
        super().__init__()
        self.embedding = nn.Embedding(100, HIDDEN)

    def forward(self, input_ids=None, attention_mask=None):
        tokens = self.embedding(input_ids)
        return type("Out", (), {"last_hidden_state": tokens})()


@pytest.fixture
def patched_encoder(monkeypatch):
    """Подменяет загрузку весов на локальный модуль."""
    monkeypatch.setattr(
        classifier_module.AutoModel, "from_pretrained", lambda *a, **k: FakeEncoder()
    )
    return FakeEncoder


def test_forward_shape_is_squeezed(patched_encoder):
    model = SpamClassifier(dropout=0.1)
    input_ids = torch.randint(0, 100, (BATCH, SEQ_LEN))
    mask = torch.ones((BATCH, SEQ_LEN), dtype=torch.long)

    out = model(input_ids, mask)
    assert out.shape == (BATCH,)
    assert not out.shape == (BATCH, 1)


def test_forward_is_float32(patched_encoder):
    model = SpamClassifier(dropout=0.1)
    out = model(torch.randint(0, 100, (BATCH, SEQ_LEN)), torch.ones((BATCH, SEQ_LEN)))
    assert out.dtype == torch.float32


def test_classifier_output_layer_is_single_logit(patched_encoder):
    model = SpamClassifier(dropout=0.1)
    assert model.classifier.out_features == 1
    assert model.classifier.in_features == HIDDEN


def test_gradients_flow_to_encoder(patched_encoder):
    """Проверяем, что энкодер действительно обучается, а не заморожен."""
    model = SpamClassifier(dropout=0.0)
    out = model(torch.randint(0, 100, (BATCH, SEQ_LEN)), torch.ones((BATCH, SEQ_LEN)))
    out.sum().backward()

    grad = model.encoder.embedding.weight.grad
    assert grad is not None
    assert grad.abs().sum() > 0


def test_classifier_head_receives_gradient(patched_encoder):
    model = SpamClassifier(dropout=0.0)
    out = model(torch.randint(0, 100, (BATCH, SEQ_LEN)), torch.ones((BATCH, SEQ_LEN)))
    out.sum().backward()
    assert model.classifier.weight.grad.abs().sum() > 0


def test_dropout_rate_is_applied(patched_encoder):
    model = SpamClassifier(dropout=0.9)
    assert model.dropout.p == 0.9


def test_eval_mode_is_deterministic(patched_encoder):
    """В eval dropout выключен — два прогона на одном батче совпадают."""
    model = SpamClassifier(dropout=0.5)
    model.eval()
    input_ids = torch.randint(0, 100, (BATCH, SEQ_LEN))
    mask = torch.ones((BATCH, SEQ_LEN), dtype=torch.long)

    first = model(input_ids, mask)
    second = model(input_ids, mask)
    assert torch.allclose(first, second)


def test_no_download_when_patched(monkeypatch):
    def explode(*args, **kwargs):
        raise AssertionError("тест попытался скачать веса из сети")

    monkeypatch.setattr(classifier_module.AutoModel, "from_pretrained", explode)
    with pytest.raises(AssertionError):
        SpamClassifier()


def test_autoconfig_is_available():
    """Проверяем, что transformers отдаёт конфиг — им пользуется SpamClassifier."""
    config = AutoConfig.for_model("bert", hidden_size=HIDDEN)
    assert config.hidden_size == HIDDEN
