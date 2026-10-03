# service/main.py
"""Точка входа сервиса: поднимает uvicorn с приложением из service.api.

Запуск:
    python -m service.main
    uvicorn service.main:app
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