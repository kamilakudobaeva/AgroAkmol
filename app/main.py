"""AgroAqkol backend — wires together the field registry, ML pipeline and AI chat layer.

Run: uvicorn app.main:app --reload   (or: make api)
"""
from fastapi import FastAPI
from app.routers import fields, predict, risk, chat

app = FastAPI(title="AgroAqkol", description="Yield forecast + drought/sukhovei/frost risk + AI chat for Akmola region fields")

app.include_router(fields.router)
app.include_router(predict.router)
app.include_router(risk.router)
app.include_router(chat.router)


@app.get("/health")
async def health() -> dict:
    return {"status": "ok"}
from fastapi.staticfiles import StaticFiles
app.mount("/", StaticFiles(directory="web", html=True), name="web")