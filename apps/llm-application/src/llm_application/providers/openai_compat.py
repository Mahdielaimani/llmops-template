"""OpenAI-compatible client. One implementation serves vLLM, Ollama, OpenAI and
anything else speaking /v1/chat/completions.

No retries, deliberately (CLAUDE.md §55): a retried LLM request duplicates GPU work
and tokens, so one timeout becomes three times the load on a saturated engine.
Retry policy belongs in the model gateway (Phase 15) where it can see queue depth.
"""

from __future__ import annotations

import json
import time
from collections.abc import AsyncIterator
from typing import Any

import httpx

from llm_application.providers.base import ChatRequest, ChatResult, Chunk, Timings, Usage
from llmops_core.errors import (
    ApiError,
    RateLimitedError,
    UpstreamTimeoutError,
    UpstreamUnavailableError,
)
from llmops_core.logging import get_logger

log = get_logger("provider.openai")

_SSE_DONE = "[DONE]"


class OpenAICompatProvider:
    def __init__(
        self,
        *,
        base_url: str,
        api_key: str | None,
        model: str,
        timeout_s: float,
        name: str = "openai_compat",
    ) -> None:
        self.name = name
        self.default_model = model
        # connect short, read generous: a slow first token is normal, an unreachable
        # host is not. `timeout_s` is the whole-request deadline.
        self._client = httpx.AsyncClient(
            base_url=base_url.rstrip("/"),
            headers={"Authorization": f"Bearer {api_key}"} if api_key else {},
            timeout=httpx.Timeout(timeout_s, connect=5.0),
        )

    def _payload(self, req: ChatRequest, *, stream: bool) -> dict[str, object]:
        body: dict[str, object] = {
            "model": req.model or self.default_model,
            "messages": [m.model_dump() for m in req.messages],
            "stream": stream,
        }
        if req.temperature is not None:
            body["temperature"] = req.temperature
        if req.max_tokens is not None:
            body["max_tokens"] = req.max_tokens
        if stream:
            # vLLM and OpenAI both report usage on the final chunk only when asked.
            body["stream_options"] = {"include_usage": True}
        return body

    async def complete(self, req: ChatRequest) -> ChatResult:
        started = time.perf_counter()
        data = await self._post("/chat/completions", self._payload(req, stream=False))
        total = (time.perf_counter() - started) * 1000
        choice = data["choices"][0]
        return ChatResult(
            content=choice["message"]["content"] or "",
            model=data.get("model", req.model or self.default_model),
            finish_reason=choice.get("finish_reason"),
            usage=_usage(data.get("usage")),
            # Non-streaming gives no first-token signal, so ttft is unknowable here.
            timings=Timings(total_ms=round(total, 2)),
        )

    async def stream(self, req: ChatRequest) -> AsyncIterator[Chunk]:
        started = time.perf_counter()
        parts: list[str] = []
        gaps: list[float] = []
        ttft: float | None = None
        last = started
        model = req.model or self.default_model
        finish_reason: str | None = None
        usage = Usage()

        try:
            async with self._client.stream(
                "POST", "/chat/completions", json=self._payload(req, stream=True)
            ) as response:
                if response.status_code >= 400:
                    await response.aread()
                    raise _map_status(response.status_code, response.text)

                async for line in response.aiter_lines():
                    if not line.startswith("data:"):
                        continue
                    payload = line[5:].strip()
                    if payload == _SSE_DONE:
                        break

                    event = json.loads(payload)
                    model = event.get("model", model)
                    if event.get("usage"):
                        usage = _usage(event["usage"])

                    choices = event.get("choices") or []
                    if not choices:
                        continue
                    finish_reason = choices[0].get("finish_reason") or finish_reason
                    delta = (choices[0].get("delta") or {}).get("content")
                    if not delta:
                        continue  # role-only opening chunk: not a token, do not time it

                    now = time.perf_counter()
                    if ttft is None:
                        ttft = (now - started) * 1000
                    else:
                        gaps.append(round((now - last) * 1000, 2))
                    last = now
                    parts.append(delta)
                    yield Chunk(delta=delta)
        except httpx.TimeoutException as exc:
            raise UpstreamTimeoutError(f"{self.name} timed out") from exc
        except httpx.HTTPError as exc:
            raise UpstreamUnavailableError(f"{self.name} unreachable") from exc

        yield Chunk(
            done=True,
            result=ChatResult(
                content="".join(parts),
                model=model,
                finish_reason=finish_reason,
                usage=usage,
                timings=Timings(
                    ttft_ms=round(ttft, 2) if ttft is not None else None,
                    total_ms=round((time.perf_counter() - started) * 1000, 2),
                    inter_token_ms=gaps,
                ),
            ),
        )

    async def _post(self, path: str, body: dict[str, object]) -> dict[str, Any]:
        try:
            response = await self._client.post(path, json=body)
        except httpx.TimeoutException as exc:
            raise UpstreamTimeoutError(f"{self.name} timed out") from exc
        except httpx.HTTPError as exc:
            raise UpstreamUnavailableError(f"{self.name} unreachable") from exc
        if response.status_code >= 400:
            raise _map_status(response.status_code, response.text)
        return response.json()  # type: ignore[no-any-return]

    async def aclose(self) -> None:
        await self._client.aclose()


def _usage(raw: dict[str, Any] | None) -> Usage:
    if not raw:
        return Usage()
    return Usage(input_tokens=raw.get("prompt_tokens"), output_tokens=raw.get("completion_tokens"))


def _map_status(status: int, body: str) -> ApiError:
    """Upstream status → our envelope. The provider's body is logged, never returned:
    it can echo the prompt back and carries no useful information for the caller."""
    log.warning("provider_error", upstream_status=status, upstream_body=body[:500])
    if status == 429:
        return RateLimitedError("upstream rate limit")
    if status in (408, 504):
        return UpstreamTimeoutError("upstream timeout")
    return UpstreamUnavailableError(f"upstream returned {status}")
