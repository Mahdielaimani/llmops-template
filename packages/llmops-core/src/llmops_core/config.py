"""Settings from the environment: LLMOPS_ prefix, __ for nesting
(LLMOPS_SERVER__PORT, LLMOPS_LLM__PROVIDER)."""

from __future__ import annotations

from enum import StrEnum
from functools import lru_cache
from typing import Literal

from pydantic import BaseModel, Field, SecretStr
from pydantic_settings import BaseSettings, SettingsConfigDict


class Environment(StrEnum):
    LOCAL = "local"
    DEV = "dev"
    STAGING = "staging"
    PROD = "prod"
    TEST = "test"


class LLMProviderKind(StrEnum):
    """mock = MODE C, vllm/ollama = MODE B (local), openai/anthropic = MODE A."""

    MOCK = "mock"
    VLLM = "vllm"
    OLLAMA = "ollama"
    OPENAI = "openai"
    ANTHROPIC = "anthropic"


class AppSettings(BaseModel):
    name: str = "llm-application"
    env: Environment = Environment.LOCAL
    log_level: Literal["DEBUG", "INFO", "WARNING", "ERROR"] = "INFO"
    log_format: Literal["json", "console"] = "json"


class ServerSettings(BaseModel):
    host: str = "0.0.0.0"  # noqa: S104 — container binds all interfaces; compose/K8s narrows it
    port: int = Field(default=8080, ge=1, le=65535)


class LLMSettings(BaseModel):
    provider: LLMProviderKind = LLMProviderKind.MOCK
    model: str = "mock-1"
    base_url: str | None = None  # OpenAI-compatible endpoint
    api_key: SecretStr | None = None
    timeout_s: float = Field(default=30.0, gt=0)
    max_tokens: int = Field(default=512, ge=1)
    temperature: float = Field(default=0.2, ge=0.0, le=2.0)


class PostgresSettings(BaseModel):
    host: str = "postgres"
    port: int = 5432
    user: str = "llmops"
    password: SecretStr = SecretStr("llmops")
    database: str = "llmops"

    @property
    def dsn(self) -> str:
        return (
            f"postgresql://{self.user}:{self.password.get_secret_value()}"
            f"@{self.host}:{self.port}/{self.database}"
        )


class RedisSettings(BaseModel):
    url: str = "redis://redis:6379/0"


class QdrantSettings(BaseModel):
    url: str = "http://qdrant:6333"
    api_key: SecretStr | None = None


class Settings(BaseSettings):
    model_config = SettingsConfigDict(
        env_prefix="LLMOPS_",
        env_nested_delimiter="__",
        env_file=".env",
        env_file_encoding="utf-8",
        extra="ignore",
        frozen=True,
    )

    app: AppSettings = AppSettings()
    server: ServerSettings = ServerSettings()
    llm: LLMSettings = LLMSettings()
    postgres: PostgresSettings = PostgresSettings()
    redis: RedisSettings = RedisSettings()
    qdrant: QdrantSettings = QdrantSettings()


@lru_cache(maxsize=1)
def get_settings() -> Settings:
    """Cached; tests must call get_settings.cache_clear() after patching the environment."""
    return Settings()
