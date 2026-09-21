from __future__ import annotations
from typing import Protocol
from .config import settings


class LLMClient(Protocol):
    async def complete(self, system: str, messages: list[dict]) -> str: ...


class OpenAICompatibleClient:
    """Работает и с OpenAI, и с любым OpenAI-совместимым API (DeepSeek, vLLM, Ollama и т.п.)."""

    def __init__(self, model: str, api_key: str, base_url: str | None):
        from openai import AsyncOpenAI
        self._client = AsyncOpenAI(api_key=api_key, base_url=base_url)
        self._model = model

    async def complete(self, system: str, messages: list[dict]) -> str:
        resp = await self._client.chat.completions.create(
            model=self._model,
            temperature=settings.temperature,
            max_tokens=settings.max_tokens,
            messages=[{"role": "system", "content": system}, *messages],
        )
        return resp.choices[0].message.content or ""


def get_llm_client() -> LLMClient:
    if settings.provider in ("openai", "deepseek", "local"):
        # DeepSeek and most local servers (vLLM, Ollama) expose an OpenAI-compatible
        # API — set CHAT_LLM_BASE_URL accordingly (e.g. https://api.deepseek.com).
        return OpenAICompatibleClient(settings.model, settings.api_key, settings.base_url)
    raise ValueError(f"Unknown LLM provider: {settings.provider}")
