"""Тесты эндпоинтов /health и /predict.

Приложение поднимается с заглушками (INBOX_CLEANER_FAKE_MODEL=1): настоящие
веса rubert (~700 МБ) в CI не скачиваются. Логика эндпоинтов при этом та же,
что в бою — отличается только содержимое runtime.
"""

import pytest
from fastapi.testclient import TestClient

from service.api import app, build_runtime
from service.stubs import fake_model_enabled

# Устройство, на котором пошла бы модель, — не часть контракта /health: ответ
# должен быть одинаковым на CI без GPU и на машине с картой. Поэтому
# проверяется форма ответа и то, что устройство названо одним из известных.
KNOWN_DEVICE_TYPES = {"cpu", "cuda", "mps", "xpu"}


@pytest.fixture
def client(monkeypatch):
    """TestClient с заглушками вместо rubert."""
    monkeypatch.setenv("INBOX_CLEANER_FAKE_MODEL", "1")
    with TestClient(app) as test_client:
        yield test_client


def test_health_reports_ok(client):
    response = client.get("/health")
    assert response.status_code == 200
    body = response.json()
    assert set(body) == {"status", "device", "model_loaded"}
    assert body["status"] == "ok"
    assert body["model_loaded"] is True
    # Индекс допустим, если он есть: "cuda:0" — тот же cuda.
    assert body["device"].split(":")[0] in KNOWN_DEVICE_TYPES


def test_predict_returns_label_and_confidence(client):
    response = client.post("/predict", json={"text": "Примите участие в розыгрыше"})
    assert response.status_code == 200
    body = response.json()
    assert body["label"] in {"spam", "ham"}
    assert 0.0 <= body["confidence"] <= 1.0


def test_predict_rejects_empty_text(client):
    """text ограничен снизу: пустая строка не проходит валидацию."""
    response = client.post("/predict", json={"text": ""})
    assert response.status_code == 422


def test_predict_rejects_text_over_limit(client):
    response = client.post("/predict", json={"text": "a" * 10001})
    assert response.status_code == 422


@pytest.mark.parametrize("threshold", [-0.1, 1.1])
def test_predict_rejects_threshold_out_of_range(client, threshold):
    response = client.post("/predict", json={"text": "привет", "threshold": threshold})
    assert response.status_code == 422


def test_predict_rejects_missing_text(client):
    response = client.post("/predict", json={})
    assert response.status_code == 422


def test_threshold_extremes_are_accepted(client):
    for threshold in (0.0, 1.0):
        response = client.post("/predict", json={"text": "привет", "threshold": threshold})
        assert response.status_code == 200


def test_urls_do_not_break_prediction(client):
    """clean_text превращает ссылки в [URL] — эндпоинт это переживает."""
    response = client.post("/predict", json={"text": "заходи http://spam.com сюда"})
    assert response.status_code == 200


def test_prediction_is_deterministic(client):
    """Модель в eval, поэтому два одинаковых письма дают одну уверенность."""
    payload = {"text": "Одинаковое письмо"}
    first = client.post("/predict", json=payload).json()
    second = client.post("/predict", json=payload).json()
    assert first["confidence"] == second["confidence"]


def test_unknown_route_returns_404(client):
    assert client.get("/nope").status_code == 404


def test_method_not_allowed(client):
    assert client.get("/predict").status_code == 405


def test_build_runtime_uses_stubs_when_flag_set(monkeypatch):
    monkeypatch.setenv("INBOX_CLEANER_FAKE_MODEL", "1")
    assert fake_model_enabled() is True
    runtime = build_runtime()
    assert runtime.model is not None
    assert runtime.tokenizer is not None


def test_fake_mode_off_by_default(monkeypatch):
    monkeypatch.delenv("INBOX_CLEANER_FAKE_MODEL", raising=False)
    assert fake_model_enabled() is False


def test_flag_values_are_strict(monkeypatch):
    """Лишние значения не включают заглушки: иначе сервер молча потеряет модель."""
    for value in ("0", "false", "no", ""):
        monkeypatch.setenv("INBOX_CLEANER_FAKE_MODEL", value)
        assert fake_model_enabled() is False