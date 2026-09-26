"""Provider contract. One interface, three modes: mock (C), local vLLM/Ollama (B),
external API (A). Callers never branch on which one is active."""

from __future__ import annotations

from collections.abc import AsyncIterator
from dataclasses import dataclass, field
from typing import Protocol

from pydantic import BaseModel, Field


class Message(BaseModel):
    role: str = Field(pattern="^(system|user|assistant)$")
    content: str = Field(min_length=1, max_length=100_000)


class ChatRequest(BaseModel):
    messages: list[Message] = Field(min_length=1)
    model: str | None = None
    temperature: float | None = Field(default=None, ge=0.0, le=2.0)
    max_tokens: int | None = Field(default=None, ge=1, le=8192)
    stream: bool = False


@dataclass
class Usage:
    """None means the provider did not report it. Never estimated — a made-up token
    count silently corrupts the cost model in docs/metrics-and-capacity.md §6."""

    input_tokens: int | None = None
    output_tokens: int | None = None

    @property
    def total(self) -> int | None:
        if self.input_tokens is None or self.output_tokens is None:
            return None
        return self.input_tokens + self.output_tokens


@dataclass
class Timings:
    """Latency breakdown. ttft_ms is measured at first *content* token, not at
    response headers — an OpenAI-compatible server sends a role-only first chunk."""

    ttft_ms: float | None = None
    total_ms: float = 0.0
    inter_token_ms: list[float] = field(default_factory=list)

    @property
    def tpot_ms(self) -> float | None:
        """Time per output token over the decode phase only, so prefill is excluded."""
        if self.ttft_ms is None or not self.inter_token_ms:
            return None
        return round((self.total_ms - self.ttft_ms) / len(self.inter_token_ms), 2)

    @property
    def tokens_per_second(self) -> float | None:
        tpot = self.tpot_ms
        return round(1000 / tpot, 2) if tpot else None


@dataclass
class ChatResult:
    content: str
    model: str
    finish_reason: str | None = None
    usage: Usage = field(default_factory=Usage)
    timings: Timings = field(default_factory=Timings)


@dataclass
class Chunk:
    """A streamed delta. The final chunk carries `done=True` plus usage and timings."""

    delta: str = ""
    done: bool = False
    result: ChatResult | None = None


class LLMProvider(Protocol):
    name: str
    default_model: str

    async def complete(self, req: ChatRequest) -> ChatResult: ...

    def stream(self, req: ChatRequest) -> AsyncIterator[Chunk]: ...

    async def aclose(self) -> None: ...
