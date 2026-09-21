from __future__ import annotations
from .schemas import YieldPrediction
from .prompts import EXPLAINER_SYSTEM

_DIRECTION_RU = {
    "down": "снизил",
    "up": "повысил",
}


def build_explanation_draft(pred: YieldPrediction) -> str:
    """Скелет объяснения без LLM: только факты из модели. Все числа в т/га."""
    low, high = pred.confidence_interval
    lines = [
        f"Прогноз урожайности: {pred.yield_t_ha:.2f} т/га "
        f"(реалистичный диапазон {low:.2f}\u2013{high:.2f} т/га), сезон {pred.season}."
    ]
    if pred.top_factors:
        lines.append("Основные факторы:")
        for f in pred.top_factors:
            verb = _DIRECTION_RU.get(f.direction, "повлиял на")
            # impact — вклад в единицах целевой переменной (т/га), знак уже учтён в direction
            lines.append(f"- {f.name}: {verb} урожайность примерно на {abs(f.impact):.2f} т/га.")
    else:
        lines.append("Значимых факторов модель не выделила.")
    return "\n".join(lines)


EXPLAINER_USER_TEMPLATE = """Данные модели (не меняй числа!):

{draft}

Переформулируй это в 3–5 предложений понятного текста для фермера.
"""


async def make_explanation(pred: YieldPrediction, llm) -> str:
    draft = build_explanation_draft(pred)
    text = await llm.complete(
        system=EXPLAINER_SYSTEM,
        messages=[{"role": "user", "content": EXPLAINER_USER_TEMPLATE.format(draft=draft)}],
    )
    return text.strip()
