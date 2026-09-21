from fastapi import APIRouter, HTTPException
from app.ai.schemas import ChatRequest, ChatResponse
from app.ai.service import chat as chat_service

router = APIRouter(tags=["chat"])


@router.post("/chat", response_model=ChatResponse)
async def chat_endpoint(body: ChatRequest) -> ChatResponse:
    try:
        reply, ctx = await chat_service(body.field_id, body.message, body.history)
    except KeyError:
        raise HTTPException(status_code=404, detail="Поле не найдено")

    return ChatResponse(reply=reply, used_context=ctx)
