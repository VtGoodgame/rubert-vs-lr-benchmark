"""Единая настройка логирования для слоёв model, service и scripts.

Слои берут логгер через get_logger(__name__) и не настраивают вывод сами.
Иначе формат сообщений разъезжается между слоями, а вывод уходит в stdout,
где его не видно в логах CI.
"""

from __future__ import annotations

import logging
import os
import sys
from typing import TextIO

DEFAULT_FORMAT = "%(asctime)s | %(levelname)-8s | %(name)-28s | %(message)s"
DEFAULT_DATE_FORMAT = "%H:%M:%S"
DEFAULT_LEVEL = "INFO"
LEVEL_ENV_VAR = "INBOX_CLEANER_LOG_LEVEL"

# Сторонние библиотеки, которые на уровне INFO засоряют вывод: каждый запрос
# к Hugging Face печатается отдельной строкой и забивает полезные сообщения.
NOISY_LOGGERS = (
    "httpx",
    "httpcore",
    "hpack",
    "urllib3",
    "filelock",
    "asyncio",
    "huggingface_hub",
    "transformers",
)


def _resolve_level(level: str | int | None) -> int:
    """Определяет уровень: явный аргумент, переменная окружения, затем дефолт."""
    if level is None:
        level = os.environ.get(LEVEL_ENV_VAR, DEFAULT_LEVEL)
    if isinstance(level, int):
        return level
    resolved = logging.getLevelNamesMapping().get(str(level).upper())
    if resolved is None:
        known = ", ".join(sorted(logging.getLevelNamesMapping()))
        raise ValueError(f"Неизвестный уровень логирования: {level!r}. Доступны: {known}")
    return resolved


def _utf8_stream(stream: TextIO) -> TextIO:
    """Переводит поток в utf-8 с заменой непечатаемых символов.

    Консоль Windows по умолчанию в cp1251 и падает на символах из писем.
    """
    reconfigure = getattr(stream, "reconfigure", None)
    if reconfigure is not None and (stream.encoding or "").lower().replace("-", "") != "utf8":
        try:
            reconfigure(encoding="utf-8", errors="replace")
        except (ValueError, OSError):
            pass
    return stream


def setup_logging(
    level: str | int | None = None,
    stream: TextIO | None = None,
    fmt: str = DEFAULT_FORMAT,
    noisy_level: int | None = logging.WARNING,
) -> logging.Logger:
    """Настраивает вывод в консоль и возвращает корневой логгер.

    Вывод идёт в stderr, а не в stdout: stdout останется чистым, и его можно
    перенаправить в файл, не смешав с логами. Повторный вызов не дублирует
    обработчики.

    Логгеры из NOISY_LOGGERS опускаются до noisy_level, иначе HTTP-запросы к
    Hugging Face забивают вывод. На уровне DEBUG приглушение не применяется:
    там нужен полный поток, иначе не разобрать причину сбоя.
    """
    resolved = _resolve_level(level)
    root = logging.getLogger()
    for handler in list(root.handlers):
        root.removeHandler(handler)

    handler = logging.StreamHandler(_utf8_stream(stream if stream is not None else sys.stderr))
    handler.setFormatter(logging.Formatter(fmt, datefmt=DEFAULT_DATE_FORMAT))
    root.addHandler(handler)
    root.setLevel(resolved)

    if noisy_level is not None and resolved > logging.DEBUG:
        for name in NOISY_LOGGERS:
            logging.getLogger(name).setLevel(noisy_level)
    return root


def get_logger(name: str) -> logging.Logger:
    """Логгер для модуля. Имя модуля попадает в формат сообщения."""
    return logging.getLogger(name)
