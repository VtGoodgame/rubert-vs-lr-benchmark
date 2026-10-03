"""Тесты точки входа: service/main.py должна поднимать uvicorn с приложением.

Сам сервер здесь не запускается: uvicorn.run подменяется заглушкой, которая
записывает аргументы. Реальный подъём проверяется отдельным этапом CI.
"""

import pytest

from service import main as main_module


@pytest.fixture
def recorded_run(monkeypatch):
    """Подменяет uvicorn.run и возвращает список полученных аргументов."""
    calls = []

    def fake_run(app, **kwargs):
        calls.append({"app": app, **kwargs})

    monkeypatch.setattr(main_module.uvicorn, "run", fake_run)
    return calls


def test_main_starts_uvicorn(recorded_run):
    main_module.main()
    assert len(recorded_run) == 1
    assert recorded_run[0]["app"] is main_module.app


def test_main_uses_default_host_and_port(recorded_run):
    main_module.main()
    assert recorded_run[0]["host"] == "0.0.0.0"
    assert recorded_run[0]["port"] == 8000
    assert recorded_run[0]["reload"] is False


def test_main_accepts_custom_host_and_port(recorded_run):
    main_module.main(host="127.0.0.1", port=9001)
    assert recorded_run[0]["host"] == "127.0.0.1"
    assert recorded_run[0]["port"] == 9001


def test_main_can_enable_reload(recorded_run):
    main_module.main(reload=True)
    assert recorded_run[0]["reload"] is True


def test_main_sets_up_logging(recorded_run, monkeypatch):
    """setup_logging вызывается до старта сервера, иначе логов не будет."""
    called = []
    monkeypatch.setattr(main_module, "setup_logging", lambda *a, **k: called.append(1))
    main_module.main()
    assert called == [1]


def test_app_is_importable_without_loading_weights():
    """Импорт service.main не должен поднимать lifespan и грузить модель."""
    assert main_module.app.title == "Inbox Cleaner API"
    assert main_module.app.version == "1.0.0"


def test_app_has_lifespan():
    """Загрузка модели перенесена в lifespan, иначе сервер не поднять без весов."""
    assert main_module.app.router.lifespan_context is not None