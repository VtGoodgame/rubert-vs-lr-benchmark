"""Тесты схем API: валидация входных данных должна отсекать мусор."""

import pytest
from pydantic import ValidationError

from common.config import SPAM_THRESHOLD
from model.api.schemas import EmailRequest, HealthResponse, SpamResponse


def test_email_request_defaults():
    """Дефолт берётся из конфига, а не зашит в схему: иначе API и метрики
    бенчмарка считались бы при разных порогах."""
    req = EmailRequest(text="привет")
    assert req.threshold == SPAM_THRESHOLD


def test_email_request_boundary_thresholds():
    assert EmailRequest(text="x", threshold=0.0).threshold == 0.0
    assert EmailRequest(text="x", threshold=1.0).threshold == 1.0


def test_email_request_threshold_out_of_range():
    with pytest.raises(ValidationError):
        EmailRequest(text="x", threshold=1.5)
    with pytest.raises(ValidationError):
        EmailRequest(text="x", threshold=-0.1)


def test_email_request_empty_text():
    with pytest.raises(ValidationError):
        EmailRequest(text="")


def test_email_request_too_long_text():
    with pytest.raises(ValidationError):
        EmailRequest(text="a" * 10001)


def test_email_request_max_length_text_ok():
    assert len(EmailRequest(text="a" * 10000).text) == 10000


def test_spam_response_serializes():
    res = SpamResponse(label="spam", confidence=0.87)
    assert res.model_dump() == {"label": "spam", "confidence": 0.87}


def test_health_response_serializes():
    res = HealthResponse(status="ok", device="cpu", model_loaded=True)
    assert res.model_dump()["model_loaded"] is True
