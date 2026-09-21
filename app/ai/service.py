from __future__ import annotations
from .schemas import YieldPrediction, RiskSummary, ChatContext
from .explainer import make_explanation
from .llm_client import get_llm_client
from .config import settings
from .security import sanitize_user_message
from .prompts import CHAT_SYSTEM

# Real ML functions from the ChatGPT-built pipeline + the field registry
# (this was a stub `from app.ml import ...` in the original AI-layer draft —
# wired here to the actual crop_yield_prediction package).
from crop_yield_prediction import load_field, predict_yield, compute_risk


async def get_yield_prediction(field_id: str) -> YieldPrediction:
    field = load_field(field_id)  # raises KeyError -> 404 at the router layer
    raw = predict_yield(field)
    return YieldPrediction(**raw)


async def get_risk_summary(field_id: str) -> RiskSummary:
    field = load_field(field_id)
    raw = compute_risk(field)
    return RiskSummary(**raw)


async def build_context(field_id: str) -> ChatContext:
    pred = await get_yield_prediction(field_id)
    risk = await get_risk_summary(field_id)
    return ChatContext(yield_prediction=pred, risk_summary=risk)


def _render_yield_block(p: YieldPrediction) -> str:
    lines = [
        f"yield_t_ha={p.yield_t_ha}",
        f"confidence_interval=[{p.confidence_interval[0]}, {p.confidence_interval[1]}]",
        f"season={p.season}",
        "top_factors:",
    ]
    for f in p.top_factors:
        lines.append(f"  - {f.name}: impact={f.impact} т/га, direction={f.direction}")
    return "\n".join(lines)


def _render_risk_block(r: RiskSummary) -> str:
    lines = [f"alert={r.alert}", "decades (risk 0.0-1.0):"]
    for d in r.decades:
        lines.append(
            f"  - {d.decade_label}: risk_score={d.risk_score}, "
            f"drought={d.drought}, sukhovei={d.sukhovei}, frost={d.frost}, note={d.note}"
        )
    return "\n".join(lines)


async def explain_prediction(field_id: str) -> str:
    pred = await get_yield_prediction(field_id)
    llm = get_llm_client()
    return await make_explanation(pred, llm)


async def chat(field_id: str, message: str, history: list) -> tuple[str, ChatContext]:
    ctx = await build_context(field_id)
    llm = get_llm_client()

    safe_message = sanitize_user_message(message, settings.max_user_message_chars)

    # ограничиваем историю
    trimmed = history[-settings.max_history_messages:]

    system = CHAT_SYSTEM.format(
        yield_prediction=_render_yield_block(ctx.yield_prediction),
        risk_summary=_render_risk_block(ctx.risk_summary),
    )

    messages = [{"role": m.role, "content": m.content} for m in trimmed]
    messages.append({"role": "user", "content": safe_message})

    reply = await llm.complete(system=system, messages=messages)
    return reply.strip(), ctx
