"""Единая настройка логирования для слоёв model, service и scripts.

Слои берут логгер через get_logger(__name__) и не настраивают вывод сами.
Иначе формат сообщений разъезжается между слоями.

Правила вывода:
- по умолчанию в терминал попадает только WARNING и выше;
- INFO видят только логгеры пакетов из INFO_PACKAGES (обучение модели);
- строка состоит из имени модуля и текста сообщения;
- файл логов не создаётся;
- длинные сообщения заменяются заглушкой, текст писем в консоль не попадает.

Сообщения пишутся по-русски, а поток не бросает UnicodeEncodeError на символах
вне кодировки консоли. Тексты чужих исключений и библиотек остаются английскими —
их пишет не мы.
"""

from __future__ import annotations

import logging
import sys
from typing import TextIO

DEFAULT_FORMAT = "%(name)s: %(message)s"
DEFAULT_LEVEL = logging.WARNING
HANDLER_NAME = "inbox-cleaner"

# INFO нужен только там, где идёт обучение: там прогресс и метрики.
INFO_PACKAGES = ("model",)

# Письмо длиннее этого в лог не попадает — вместо текста печатается заглушка.
MAX_MESSAGE_CHARS = 1000


class SuppressLongMessages(logging.Filter):
    """Отсекает текст писем: длинное сообщение заменяется заглушкой."""

    def filter(self, record: logging.LogRecord) -> bool:
        try:
            message = record.getMessage()
        except Exception:
            return True
        if len(message) <= MAX_MESSAGE_CHARS:
            return True
        record.msg = f"длинное сообщение скрыто ({len(message)} символов)"
        record.args = ()
        return True


def _safe_stream(stream: TextIO) -> TextIO:
    """Запрещает падение на символах вне кодировки консоли.

    Кодировку потока не меняем: в консоли Windows это cp866 или cp1251, и utf-8
    превратил бы русский текст в нечитаемую мешанину. Через errors="replace"
    неизвестный символ заменяется на "?" вместо UnicodeEncodeError.
    """
    reconfigure = getattr(stream, "reconfigure", None)
    if reconfigure is not None:
        reconfigure(errors="replace")
    return stream


def setup_logging(
    level: int = DEFAULT_LEVEL,
    stream: TextIO | None = None,
) -> logging.Logger:
    """Настраивает вывод в консоль и возвращает корневой логгер.

    Вывод идёт в stderr, чтобы stdout остался чистым. Повторный вызов
    заменяет наш обработчик, а не добавляет второй.
    """
    root = logging.getLogger()
    root.setLevel(level)

    for handler in list(root.handlers):
        if handler.get_name() == HANDLER_NAME:
            root.removeHandler(handler)
            handler.close()

    handler = logging.StreamHandler(_safe_stream(sys.stderr if stream is None else stream))
    handler.set_name(HANDLER_NAME)
    handler.setFormatter(logging.Formatter(DEFAULT_FORMAT))
    handler.addFilter(SuppressLongMessages())
    root.addHandler(handler)

    for name in INFO_PACKAGES:
        logging.getLogger(name).setLevel(logging.INFO)
    return root


def get_logger(name: str) -> logging.Logger:
    """Логгер для модуля. Имя модуля попадает в формат сообщения."""
    return logging.getLogger(name)
