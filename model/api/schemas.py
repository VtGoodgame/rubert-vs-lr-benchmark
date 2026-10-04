# model/api/schemas.py
"""Pydantic-схемы запроса и ответа API.

Порог по умолчанию берётся из common/config.py, а не задаётся здесь числом:
иначе значение разъехалось бы с тем, что подобрано в бенчмарке.
"""

from pydantic import BaseModel, Field

from common.config import SPAM_THRESHOLD


class EmailRequest(BaseModel):
    text: str = Field(min_length=1, max_length=10000,
                      description="Текст письма для классификации")
    threshold: float = Field(default=SPAM_THRESHOLD, ge=0.0, le=1.0,
                             description="Порог spam/ham; по умолчанию рабочая "
                                         "точка обслуживаемой модели")


class SpamResponse(BaseModel):
    label: str
    confidence: float


class HealthResponse(BaseModel):
    status: str
    device: str
    model_loaded: bool
