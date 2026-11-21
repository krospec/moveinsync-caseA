"""Central configuration.

Everything is environment-driven so the same image runs in docker compose,
in tests (fake LLM, frozen clock, sqlite) and in a local dev shell.
"""
from functools import lru_cache

from pydantic_settings import BaseSettings, SettingsConfigDict


class Settings(BaseSettings):
    model_config = SettingsConfigDict(
        env_file=".env", env_file_encoding="utf-8", extra="ignore"
    )

    # --- app ---
    app_name: str = "MoveInSync Shift-Change Request Agent"
    timezone: str = "Asia/Kolkata"
    # ISO datetime; when set, the whole app runs on a frozen clock.
    # Used by tests and the evaluation runner so cut-off rules are deterministic.
    frozen_now: str | None = None

    # --- persistence ---
    # Postgres in production/docker compose; sqlite only as a dev/test fallback.
    database_url: str = "sqlite+aiosqlite:///./dev.db"

    # --- auth (JWT) ---
    jwt_secret: str = "dev-secret-change-me"
    jwt_algorithm: str = "HS256"
    jwt_ttl_minutes: int = 720

    # --- llm ---
    # groq (default, free tier, OpenAI-compatible) | openai | gemini | fake
    llm_provider: str = "groq"
    llm_model: str | None = None
    groq_api_key: str | None = None
    groq_base_url: str = "https://api.groq.com/openai/v1"
    openai_api_key: str | None = None
    gemini_api_key: str | None = None
    # Hard requirement from the case study: reply within 5s p95 including the
    # LLM call, and at most 4 LLM calls per employee message.
    llm_timeout_s: float = 3.5
    llm_max_calls_per_turn: int = 4
    llm_max_output_tokens: int = 300
    llm_temperature: float = 0.0
    history_window: int = 12


@lru_cache
def get_settings() -> Settings:
    return Settings()
