# service/api.py
"""HTTP-интерфейс классификатора писем.

Модель загружается в lifespan, а не на уровне импорта: иначе модуль нельзя
импортировать в тестах, а сервер нельзя поднять без обученного чекпоинта.
Состояние лежит в app.state, поэтому тесты подставляют заглушки через
переопределение get_runtime.
"""

from contextlib import asynccontextmanager

import torch
from fastapi import FastAPI, Request

from common.config import CHECKPOINT_PATH, MAX_LENGTH, MODEL_NAME
from common.logging_setup import get_logger
from model.api import schemas
from model.preprocessing.clean import clean_text
from model.spam_classifier import SpamClassifier
from model.tokenization.tokenizer import TextTokenizer
from service.stubs import TinyTokenizer, build_tiny_model, fake_model_enabled

logger = get_logger(__name__)


class Runtime:
    """Всё, что нужно эндпоинтам: токенизатор, модель и устройство."""

    def __init__(self, tokenizer, model, device):
        self.tokenizer = tokenizer
        self.model = model
        self.device = device


def build_runtime() -> Runtime:
    """Собирает токенизатор и модель.

    При INBOX_CLEANER_FAKE_MODEL=1 берутся заглушки: это позволяет поднять
    сервер в CI без весов rubert и без сети.
    """
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    if fake_model_enabled():
        logger.warning("режим заглушек: настоящие веса не загружаются")
        # Заглушку тоже нужно перенести на устройство. Раньше здесь стоял
        # голый build_tiny_model(), и на машине с GPU /predict падал:
        # токенизатор отдавал тензоры на CPU, predict() переносил их на cuda,
        # а веса микроэнкодера оставались на CPU. На CI этого не видно, потому
        # что там устройство всегда одно.
        return Runtime(TinyTokenizer(max_length=8), build_tiny_model().to(device), device)

    logger.info("загружаю веса из %s", CHECKPOINT_PATH)
    tokenizer = TextTokenizer(model_name=MODEL_NAME, max_length=MAX_LENGTH)
    model = SpamClassifier(model_name=MODEL_NAME)
    model.load_state_dict(torch.load(CHECKPOINT_PATH, weights_only=True))
    model.to(device)
    model.eval()
    logger.info("модель загружена на устройство %s", device)
    return Runtime(tokenizer, model, device)


def get_runtime(request: Request) -> Runtime:
    """Достаёт runtime из состояния приложения."""
    return request.app.state.runtime


@asynccontextmanager
async def lifespan(application: FastAPI):
    application.state.runtime = build_runtime()
    yield


app = FastAPI(title="Inbox Cleaner API", version="1.0.0", lifespan=lifespan)


@app.get("/health", response_model=schemas.HealthResponse)
def health(request: Request):
    """Проверка, что API жив и модель загружена."""
    runtime = get_runtime(request)
    return schemas.HealthResponse(
        status="ok",
        device=str(runtime.device),
        model_loaded=runtime.model is not None,
    )


@app.post("/predict", response_model=schemas.SpamResponse)
def predict(request: Request, email: schemas.EmailRequest):
    """Классификация письма: spam или ham."""
    runtime = get_runtime(request)
    cleaned = clean_text(email.text)
    enc = runtime.tokenizer(cleaned)
    input_ids = enc["input_ids"].to(runtime.device)
    attention_mask = enc["attention_mask"].to(runtime.device)

    with torch.no_grad():
        logit = runtime.model(input_ids, attention_mask)
        prob = torch.sigmoid(logit).item()

    label = "spam" if prob > email.threshold else "ham"
    logger.info("label=%s confidence=%.4f threshold=%.2f", label, prob, email.threshold)
    return schemas.SpamResponse(label=label, confidence=prob)