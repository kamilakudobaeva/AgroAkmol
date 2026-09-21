from pydantic_settings import BaseSettings, SettingsConfigDict


class AISettings(BaseSettings):
    model_config = SettingsConfigDict(env_prefix="CHAT_LLM_", env_file=".env")

    provider: str = "openai"          # openai | anthropic | local | deepseek (OpenAI-compatible base_url)
    model: str = "gpt-4o-mini"
    api_key: str = ""
    base_url: str | None = None
    temperature: float = 0.2
    max_tokens: int = 800

    # лимиты
    max_history_messages: int = 10
    max_user_message_chars: int = 2000


settings = AISettings()
