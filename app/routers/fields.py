from fastapi import APIRouter
from app.ai.schemas import FieldCreateRequest, FieldCreateResponse
from crop_yield_prediction import create_field

router = APIRouter(tags=["fields"])


@router.post("/fields", response_model=FieldCreateResponse)
async def register_field(body: FieldCreateRequest) -> FieldCreateResponse:
    field_id = create_field(
        name=body.name, district=body.district, latitude=body.latitude,
        longitude=body.longitude, crop=body.crop, area_ha=body.area_ha,
    )
    return FieldCreateResponse(field_id=field_id)
