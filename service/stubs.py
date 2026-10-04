"""Заглушки модели и токенизатора для тестов и CI.

Нужны, чтобы поднять сервер и прогнать эндпоинты без весов rubert (~700 МБ)
и без обращения к сети. В боевом режиме не используются: там в service/api.py
подставляются настоящие TextTokenizer и SpamClassifier.

Включаются переменной окружения INBOX_CLEANER_FAKE_MODEL=1.
"""

import os

import torch
import torch.nn as nn

FAKE_MODEL_ENV = "INBOX_CLEANER_FAKE_MODEL"
VOCAB_SIZE = 128
HIDDEN_SIZE = 16
SEQ_LEN = 8


def fake_model_enabled() -> bool:
    """Включён ли режим заглушек."""
    return os.environ.get(FAKE_MODEL_ENV, "").strip() in {"1", "true", "True"}


class TinyEncoder(nn.Module):
    """Энкодер нужной формы: возвращает [batch, seq_len, hidden]."""

    class _Config:
        hidden_size = HIDDEN_SIZE

    config = _Config()

    def __init__(self):
        super().__init__()
        self.embedding = nn.Embedding(VOCAB_SIZE, HIDDEN_SIZE)

    def forward(self, input_ids=None, attention_mask=None):
        tokens = self.embedding(input_ids)
        return type("EncoderOutput", (), {"last_hidden_state": tokens})()


class TinyTokenizer:
    """Токенизатор без сети: символы кодируются по модулю размера словаря.

    Возвращает те же ключи, что и TextTokenizer, поэтому эндпоинт /predict
    проверяется тем же кодом, что и в бою.
    """

    def __init__(self, max_length: int = SEQ_LEN):
        self.max_length = max_length

    def _encode(self, text: str) -> torch.Tensor:
        ids = [ord(ch) % VOCAB_SIZE for ch in text[: self.max_length]]
        ids = ids + [0] * (self.max_length - len(ids))
        return torch.tensor([ids], dtype=torch.long)

    def __call__(self, text: str) -> dict:
        input_ids = self._encode(text)
        attention_mask = torch.ones_like(input_ids)
        attention_mask[0, 0] = 1
        return {"input_ids": input_ids, "attention_mask": attention_mask}

    def batch(self, texts: list[str]) -> dict:
        encoded = [self(text) for text in texts]
        return {
            "input_ids": torch.cat([item["input_ids"] for item in encoded]),
            "attention_mask": torch.cat([item["attention_mask"] for item in encoded]),
        }

    @property
    def vocab_size(self) -> int:
        return VOCAB_SIZE


def build_tiny_model():
    """Готовая модель для тестов: классификатор с крошечным энкодером."""
    from model.spam_classifier import SpamClassifier

    model = SpamClassifier(dropout=0.0, encoder=TinyEncoder())
    model.eval()
    return model