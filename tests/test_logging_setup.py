"""Тесты настройки логирования.

Проверяем правила из README: без дублей обработчиков, INFO только у пакета
model, длинные тексты писем заменяются заглушкой.
"""

import io
import logging

import pytest

from common.logging_setup import (
    DEFAULT_LEVEL,
    HANDLER_NAME,
    MAX_MESSAGE_CHARS,
    get_logger,
    setup_logging,
)


@pytest.fixture
def stream():
    """Свой поток вместо stderr — чтобы читать вывод теста."""
    return io.StringIO()


@pytest.fixture
def configured(stream):
    """Настраивает логирование и убирает обработчики после теста."""
    setup_logging(stream=stream)
    yield stream
    root = logging.getLogger()
    for handler in list(root.handlers):
        if handler.get_name() == HANDLER_NAME:
            root.removeHandler(handler)
            handler.close()


def test_default_level_is_warning():
    assert DEFAULT_LEVEL == logging.WARNING


def test_repeated_setup_does_not_duplicate_handlers(stream):
    setup_logging(stream=stream)
    setup_logging(stream=stream)
    setup_logging(stream=stream)
    root = logging.getLogger()
    ours = [h for h in root.handlers if h.get_name() == HANDLER_NAME]
    assert len(ours) == 1


def test_info_hidden_by_default(configured):
    logging.getLogger("service.api").info("не должно попасть в вывод")
    assert configured.getvalue() == ""


def test_info_visible_for_model_package(configured):
    logging.getLogger("model.train").info("видно обучение")
    assert "видно обучение" in configured.getvalue()


def test_warning_is_always_visible(configured):
    logging.getLogger("service.api").warning("предупреждение видно")
    assert "предупреждение видно" in configured.getvalue()


def test_format_contains_module_name(configured):
    logging.getLogger("model.train").warning("сообщение")
    output = configured.getvalue()
    assert "model.train" in output
    assert "сообщение" in output


def test_long_message_replaced_with_stub(configured):
    """Текст письма длиннее MAX_MESSAGE_CHARS в консоль не попадает."""
    letter = "спам " * 500
    assert len(letter) > MAX_MESSAGE_CHARS
    logging.getLogger("model.train").info(letter)
    output = configured.getvalue()
    assert letter not in output
    assert "длинное сообщение скрыто" in output


def test_short_message_not_suppressed(configured):
    logging.getLogger("model.train").info("короткое сообщение")
    assert "короткое сообщение" in configured.getvalue()


def test_foreign_handlers_untouched(stream):
    root = logging.getLogger()
    foreign = logging.StreamHandler(stream)
    foreign.set_name("чужой-обработчик")
    root.addHandler(foreign)
    try:
        setup_logging(stream=stream)
        names = [h.get_name() for h in root.handlers]
        assert "чужой-обработчик" in names
    finally:
        root.removeHandler(foreign)


def test_get_logger_returns_named_logger():
    assert get_logger("model.train").name == "model.train"
