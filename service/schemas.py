"""Контракты HTTP-слоя."""

from pydantic import BaseModel, Field


class PredictRequest(BaseModel):
    texts: list[str] = Field(min_length=1, max_length=100)


class PredictResponse(BaseModel):
    labels: list[str]
