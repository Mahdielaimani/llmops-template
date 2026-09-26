from __future__ import annotations

from llm_application.providers.base import (
    ChatRequest,
    ChatResult,
    Chunk,
    LLMProvider,
    Message,
    Timings,
    Usage,
)
from llm_application.providers.mock import MockProvider
from llm_application.providers.openai_compat import OpenAICompatProvider
from llmops_core.config import LLMProviderKind, Settings

_DEFAULT_BASE_URL = {
    LLMProviderKind.OPENAI: "https://api.openai.com/v1",
    LLMProviderKind.VLLM: "http://vllm:8000/v1",
    LLMProviderKind.OLLAMA: "http://host.docker.internal:11434/v1",
}


def build_provider(settings: Settings) -> LLMProvider:
    kind = settings.llm.provider
    if kind is LLMProviderKind.MOCK:
        return MockProvider(model=settings.llm.model)
    if kind is LLMProviderKind.ANTHROPIC:
        # Anthropic's Messages API is not OpenAI-shaped; a native client lands with
        # the model gateway in Phase 15.
        raise NotImplementedError("anthropic provider arrives with the model gateway")

    base_url = settings.llm.base_url or _DEFAULT_BASE_URL[kind]
    return OpenAICompatProvider(
        base_url=base_url,
        api_key=settings.llm.api_key.get_secret_value() if settings.llm.api_key else None,
        model=settings.llm.model,
        timeout_s=settings.llm.timeout_s,
        name=kind.value,
    )


__all__ = [
    "ChatRequest",
    "ChatResult",
    "Chunk",
    "LLMProvider",
    "Message",
    "MockProvider",
    "OpenAICompatProvider",
    "Timings",
    "Usage",
    "build_provider",
]
