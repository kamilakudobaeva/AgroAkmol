from fastapi import APIRouter, HTTPException
from app.ai.schemas import RiskSummary
from app.ai.service import get_risk_summary

router = APIRouter(tags=["risk"])


@router.get("/risk/{field_id}", response_model=RiskSummary)
async def risk(field_id: str) -> RiskSummary:
    try:
        return await get_risk_summary(field_id)
    except KeyError:
        raise HTTPException(status_code=404, detail="Поле не найдено")
