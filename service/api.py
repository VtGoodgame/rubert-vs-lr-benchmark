# inbox-cleaner/api.py
import torch
from fastapi import FastAPI

from common.logging_setup import get_logger, setup_logging
from model.api import schemas
from model.preprocessing.clean import clean_text
from model.spam_classifier import SpamClassifier
from model.tokenization.tokenizer import TextTokenizer

logger = get_logger(__name__)

setup_logging()

app = FastAPI(title="Inbox Cleaner API", version="1.0.0")

#Загрузка модели один раз при старте
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
tok = TextTokenizer(model_name="DeepPavlov/rubert-base-cased", max_length=128)

logger.info("загружаю веса из model/checkpoints/best.pt")
model = SpamClassifier(model_name="DeepPavlov/rubert-base-cased")
model.load_state_dict(torch.load("model/checkpoints/best.pt", weights_only=True))
model.to(device)
model.eval()
logger.info("модель загружена на устройство %s", device)

@app.get("/health", response_model=schemas.HealthResponse)
def health():
    """Проверка, что API жив и модель загружена."""
    return schemas.HealthResponse(
        status="ok",
        device=str(device),
        model_loaded=model is not None
    )

@app.post("/predict", response_model=schemas.SpamResponse)
def predict(request: schemas.EmailRequest):
    """Классификация письма: spam или ham."""
    cleaned = clean_text(request.text)
    enc = tok(cleaned)
    input_ids = enc["input_ids"].to(device)
    attention_mask = enc["attention_mask"].to(device)

    with torch.no_grad():
        logit = model(input_ids, attention_mask)
        prob = torch.sigmoid(logit).item()

    label = "spam" if prob > request.threshold else "ham"
    logger.info("label=%s confidence=%.4f threshold=%.2f", label, prob, request.threshold)
    return schemas.SpamResponse(label=label, confidence=prob)