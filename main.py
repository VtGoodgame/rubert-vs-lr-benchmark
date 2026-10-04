"""Точка входа приложения: поднимает uvicorn с приложением из service.api.

Файл лежит в корне репозитория, а не внутри пакета service: точка входа
приложения всегда должна быть на верхнем уровне, чтобы запуск не зависел
от внутреннего устройства пакета.

Запуск:
    python main.py
    uvicorn main:app
"""

import uvicorn

from common.logging_setup import setup_logging
from service.api import app

HOST = "0.0.0.0"
PORT = 8000


def main(host: str = HOST, port: int = PORT, reload: bool = False) -> None:
    """Поднимает HTTP-сервер и блокирует до остановки."""
    setup_logging()
    uvicorn.run(app, host=host, port=port, reload=reload)


if __name__ == "__main__":
    main()