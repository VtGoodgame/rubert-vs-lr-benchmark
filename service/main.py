"""Точка входа API: uvicorn service.main:app --reload."""

from fastapi import FastAPI, HTTPException

from model.predict import predict
from service.schemas import PredictRequest, PredictResponse

app = FastAPI(title="inbox-cleaner", version="0.1.0")


@app.get("/health")
def health() -> dict[str, str]:
    return {"status": "ok"}


@app.post("/predict", response_model=PredictResponse)
def predict_handler(request: PredictRequest) -> PredictResponse:
    try:
        labels = predict(request.texts)
    except FileNotFoundError as exc:
        raise HTTPException(status_code=503, detail=str(exc)) from exc
    return PredictResponse(labels=labels)
