"""Слой service: HTTP-обёртка над model. Без логики обучения."""

from service.schemas import PredictRequest, PredictResponse

__all__ = ["PredictRequest", "PredictResponse"]
