from __future__ import annotations
from typing import Literal, Optional
from pydantic import BaseModel, Field


# ---------- Сырые данные от ML ----------

class TopFactor(BaseModel):
    name: str
    impact: float
    direction: Literal["up", "down"]  # contract fix: mobile app expects exactly these two values


class YieldPrediction(BaseModel):
    yield_t_ha: float
    confidence_interval: tuple[float, float]  # [low, high]
    top_factors: list[TopFactor]
    season: str


class DecadeRisk(BaseModel):
    decade_label: str
    risk_score: float
    drought: float
    sukhovei: float
    frost: float
    note: str


class RiskSummary(BaseModel):
    decades: list[DecadeRisk]
    alert: bool


# ---------- /predict/{field_id} ----------

class PredictResponse(BaseModel):
    field_id: str
    yield_t_ha: float
    confidence_interval: tuple[float, float]
    top_factors: list[TopFactor]
    season: str
    explanation: str = Field(..., description="Объяснение на русском, без изменения чисел")


# ---------- /chat ----------

class ChatMessage(BaseModel):
    role: Literal["user", "assistant"]
    content: str


class ChatRequest(BaseModel):
    field_id: str
    message: str
    history: list[ChatMessage] = Field(default_factory=list)


class ChatContext(BaseModel):
    yield_prediction: YieldPrediction
    risk_summary: RiskSummary


class ChatResponse(BaseModel):
    reply: str
    used_context: ChatContext


# ---------- POST /fields ----------

class FieldCreateRequest(BaseModel):
    name: str
    district: str
    latitude: float
    longitude: float
    crop: str
    area_ha: float


class FieldCreateResponse(BaseModel):
    field_id: str
