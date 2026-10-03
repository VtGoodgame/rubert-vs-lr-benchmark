"""Общие фикстуры тестов.

Правило: тесты не ходят в сеть и не читают реальные данные из model/data/.
Всё живёт в tmp_path, а вместо настоящих моделей HuggingFace — заглушки,
поэтому CI не скачивает веса rubert.
"""

import sys
from pathlib import Path

import pandas as pd
import pytest
import torch

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

MAX_LENGTH = 8
VOCAB_SIZE = 100


class FakeTokenizer:
    """Заглушка вместо transformers.AutoTokenizer.

    Возвращает тензоры нужной формы, чтобы тесты проверяли форму данных,
    а не поведение настоящего токенизатора.
    """

    def __init__(self, max_length: int = MAX_LENGTH):
        self.max_length = max_length
        self.calls: list[dict] = []
        self._vocab_size = VOCAB_SIZE

    def __call__(self, text, **kwargs):
        return self._encode([text], kwargs)

    def batch(self, texts):
        return self._encode(texts, {})

    def _encode(self, texts, kwargs):
        self.calls.append({"texts": list(texts), "kwargs": kwargs})
        n = len(texts)
        length = self.max_length
        input_ids = torch.arange(n * length, dtype=torch.long).reshape(n, length) % VOCAB_SIZE
        attention_mask = torch.ones((n, length), dtype=torch.long)
        return {"input_ids": input_ids, "attention_mask": attention_mask}

    @property
    def vocab_size(self) -> int:
        return self._vocab_size


class FakeBatch(dict):
    """Батч в форме dict — так его отдаёт настоящий DataLoader."""


def make_batch(input_ids, attention_mask, labels):
    """Собирает батч из готовых тензоров."""
    return FakeBatch(
        input_ids=torch.as_tensor(input_ids),
        attention_mask=torch.as_tensor(attention_mask),
        labels=torch.as_tensor(labels, dtype=torch.float32),
    )


class FakeModel:
    """Модель с заранее заданными логитами.

    evaluate() ждёт logits, а вероятность получается через sigmoid,
    поэтому достаточно подставить логит: чем он больше, тем выше шанс spam.

    Логиты выдаются по размеру батча: если размер не совпал, берётся столько,
    сколько просил батч, иначе evaluate() упал бы на склейке меток.
    """

    def __init__(self, logits):
        self.logits = torch.as_tensor(logits, dtype=torch.float32)

    def eval(self):
        return self

    def __call__(self, input_ids, attention_mask):
        batch_size = input_ids.shape[0]
        if batch_size == len(self.logits):
            return self.logits
        if len(self.logits) % batch_size == 0:
            per_batch = len(self.logits) // batch_size
            return self.logits[:per_batch]
        return self.logits[:batch_size]


@pytest.fixture
def fake_tokenizer():
    return FakeTokenizer()


@pytest.fixture
def tmp_csv(tmp_path):
    """CSV в стиле model/data: колонки target и text."""
    def _make(rows, name="data.csv"):
        path = tmp_path / name
        pd.DataFrame(rows, columns=["target", "text"]).to_csv(path, index=False)
        return path
    return _make


@pytest.fixture
def spam_ham_rows():
    """Детерминированные строки с метками — 4 spam и 4 ham."""
    return [
        {"target": "spam", "text": "выигрыш миллион кликни сюда"},
        {"target": "ham", "text": "встреча завтра в десять"},
        {"target": "spam", "text": "бесплатно получите приз прямо сейчас"},
        {"target": "ham", "text": "отправил отчёт по проекту"},
        {"target": "spam", "text": "срочно переведите деньги на карту"},
        {"target": "ham", "text": "спасибо за помощь"},
        {"target": "spam", "text": "выиграли в лотерею звоните немедленно"},
        {"target": "ham", "text": "обед в кафе в полдень"},
    ]
