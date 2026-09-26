"""Deterministic provider for tests, CI and any phase where the GPU is busy.

Simulates the shape of real inference — a prefill delay then per-token delays — so
that streaming, timeout and backpressure code paths are exercised without a model.
Defaults are near-zero; tests that care about timing set them explicitly.
"""

from __future__ import annotations

import asyncio
import time
from collections.abc import AsyncIterator

from llm_application.providers.base import ChatRequest, ChatResult, Chunk, Timings, Usage

_WORDS_PER_TOKEN = 0.75  # rough English ratio, used only for the mock's fake usage


class MockProvider:
    name = "mock"
    default_model = "mock-1"

    def __init__(
        self, *, model: str = "mock-1", prefill_s: float = 0.0, per_token_s: float = 0.0
    ) -> None:
        self.default_model = model
        self.prefill_s = prefill_s
        self.per_token_s = per_token_s

    def _reply(self, req: ChatRequest) -> list[str]:
        last = req.messages[-1].content
        text = f"Mock reply to: {last[:200]}"
        limit = req.max_tokens or 512
        return text.split(" ")[:limit]

    async def complete(self, req: ChatRequest) -> ChatResult:
        started = time.perf_counter()
        tokens = self._reply(req)
        await asyncio.sleep(self.prefill_s)
        ttft = (time.perf_counter() - started) * 1000
        if self.per_token_s:
            await asyncio.sleep(self.per_token_s * len(tokens))
        total = (time.perf_counter() - started) * 1000
        return ChatResult(
            content=" ".join(tokens),
            model=req.model or self.default_model,
            finish_reason="stop",
            usage=self._usage(req, tokens),
            timings=Timings(
                ttft_ms=round(ttft, 2),
                total_ms=round(total, 2),
                inter_token_ms=[self.per_token_s * 1000] * max(len(tokens) - 1, 0),
            ),
        )

    async def stream(self, req: ChatRequest) -> AsyncIterator[Chunk]:
        started = time.perf_counter()
        tokens = self._reply(req)
        await asyncio.sleep(self.prefill_s)

        ttft: float | None = None
        gaps: list[float] = []
        last = started
        for i, tok in enumerate(tokens):
            if self.per_token_s:
                await asyncio.sleep(self.per_token_s)
            now = time.perf_counter()
            if ttft is None:
                ttft = (now - started) * 1000
            else:
                gaps.append(round((now - last) * 1000, 2))
            last = now
            yield Chunk(delta=tok if i == 0 else f" {tok}")

        yield Chunk(
            done=True,
            result=ChatResult(
                content=" ".join(tokens),
                model=req.model or self.default_model,
                finish_reason="stop",
                usage=self._usage(req, tokens),
                timings=Timings(
                    ttft_ms=round(ttft, 2) if ttft else None,
                    total_ms=round((time.perf_counter() - started) * 1000, 2),
                    inter_token_ms=gaps,
                ),
            ),
        )

    def _usage(self, req: ChatRequest, tokens: list[str]) -> Usage:
        prompt_words = sum(len(m.content.split()) for m in req.messages)
        return Usage(
            input_tokens=int(prompt_words / _WORDS_PER_TOKEN),
            output_tokens=len(tokens),
        )

    async def aclose(self) -> None:
        return None
