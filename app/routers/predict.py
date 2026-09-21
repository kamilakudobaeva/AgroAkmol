from fastapi import APIRouter, HTTPException
from app.ai.schemas import PredictResponse
from app.ai.service import get_yield_prediction, explain_prediction

router = APIRouter(tags=["predict"])


@router.get("/predict/{field_id}", response_model=PredictResponse)
async def predict(field_id: str) -> PredictResponse:
    try:
        pred = await get_yield_prediction(field_id)
    except KeyError:
        raise HTTPException(status_code=404, detail="Поле не найдено")

    explanation = await explain_prediction(field_id)

    return PredictResponse(
        field_id=field_id,
        yield_t_ha=pred.yield_t_ha,
        confidence_interval=pred.confidence_interval,
        top_factors=pred.top_factors,
        season=pred.season,
        explanation=explanation,
    )
