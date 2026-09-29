from pydantic import BaseModel, Field

class EmailRequest(BaseModel):
    text: str = Field(min_length=1, max_length=10000,
                      description="Текст письма для классификации")
    threshold: float = Field(default=0.5, ge=0.0, le=1.0,
                             description="Порог spam/ham")

class SpamResponse(BaseModel):
    label: str
    confidence: float

class HealthResponse(BaseModel):
    status: str
    device: str
    model_loaded: bool
