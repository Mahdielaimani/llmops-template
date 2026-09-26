"""POST /chat — the Layer 5 entry point.

Log field names follow the OpenTelemetry GenAI semantic conventions
(`gen_ai.request.model`, `gen_ai.usage.*`) so Phase 22 dashboards and any
OTel-aware backend read them without a translation layer. TTFT, TPOT and ITL have
no semconv equivalent and stay under `llm.*`; see docs/adr/ADR-017.
"""

from __future__ import annotations

import asyncio
import json
from collections.abc import AsyncIterator
from typing import Any

from fastapi import APIRouter, Request
from fastapi.responses import StreamingResponse
from pydantic import BaseModel

from llm_application.providers import ChatRequest, ChatResult, LLMProvider
from llmops_core.context import get_request_id
from llmops_core.errors import UpstreamTimeoutError
from llmops_core.logging import get_logger

log = get_logger("chat")
router = APIRouter(tags=["chat"])


class ChatResponse(BaseModel):
    answer: str
    request_id: str | None
    model: str
    finish_reason: str | None
    latency_ms: float
    ttft_ms: float | None
    tpot_ms: float | None
    tokens_per_second: float | None
    input_tokens: int | None
    output_tokens: int | None
    total_tokens: int | None
    grounded: bool = False  # /chat is ungrounded by contract; /rag (Phase 8) sets True


def _telemetry(result: ChatResult) -> dict[str, Any]:
    t = result.timings
    return {
        "gen_ai.request.model": result.model,
        "gen_ai.response.finish_reason": result.finish_reason,
        "gen_ai.usage.input_tokens": result.usage.input_tokens,
        "gen_ai.usage.output_tokens": result.usage.output_tokens,
        "llm.ttft_ms": t.ttft_ms,
        "llm.tpot_ms": t.tpot_ms,
        "llm.tokens_per_second": t.tokens_per_second,
        "llm.total_ms": t.total_ms,
    }


def _to_response(result: ChatResult) -> ChatResponse:
    return ChatResponse(
        answer=result.content,
        request_id=get_request_id(),
        model=result.model,
        finish_reason=result.finish_reason,
        latency_ms=result.timings.total_ms,
        ttft_ms=result.timings.ttft_ms,
        tpot_ms=result.timings.tpot_ms,
        tokens_per_second=result.timings.tokens_per_second,
        input_tokens=result.usage.input_tokens,
        output_tokens=result.usage.output_tokens,
        total_tokens=result.usage.total,
    )


@router.post("/chat", response_model=None)
async def chat(req: ChatRequest, request: Request) -> ChatResponse | StreamingResponse:
    provider: LLMProvider = request.app.state.provider
    deadline = request.app.state.settings.llm.timeout_s

    if not req.stream:
        result = await asyncio.wait_for(provider.complete(req), timeout=deadline)
        log.info("chat", stream=False, **_telemetry(result))
        return _to_response(result)

    return StreamingResponse(
        _sse(provider, req, deadline),
        media_type="text/event-stream",
        # Kong and any intermediate proxy must not buffer, or TTFT becomes total latency.
        headers={"Cache-Control": "no-cache", "X-Accel-Buffering": "no"},
    )


async def _sse(provider: LLMProvider, req: ChatRequest, deadline: float) -> AsyncIterator[str]:
    """Server-sent events. The deadline covers the whole stream, not each chunk —
    a provider that emits one token per minute forever would otherwise never trip it."""
    loop = asyncio.get_running_loop()
    expires_at = loop.time() + deadline
    try:
        async for chunk in provider.stream(req):
            if loop.time() > expires_at:
                raise UpstreamTimeoutError(f"stream exceeded {deadline}s")
            if chunk.done and chunk.result:
                log.info("chat", stream=True, **_telemetry(chunk.result))
                yield _event("done", _to_response(chunk.result).model_dump())
                return
            yield _event("delta", {"delta": chunk.delta})
    except Exception as exc:
        # Headers are already sent, so the error envelope cannot be an HTTP status.
        # It travels as a terminal event instead; clients must handle `event: error`.
        code = getattr(exc, "code", "internal_error")
        log.warning("chat_stream_failed", error_code=code, exc_type=type(exc).__name__)
        yield _event("error", {"error": {"code": code, "request_id": get_request_id()}})


def _event(name: str, data: dict[str, Any]) -> str:
    return f"event: {name}\ndata: {json.dumps(data)}\n\n"
