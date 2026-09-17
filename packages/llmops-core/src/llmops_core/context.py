"""Request-scoped context: request_id propagation.

The ``X-Request-ID`` header is the correlation key across gateway → app →
model-gateway → inference. In Phase 16 Kong's ``correlation-id`` plugin
generates it at the edge; until then the app generates it itself.

Implementation is ASGI-level (not ``BaseHTTPMiddleware``) so it works
correctly with streaming responses (SSE) added in Phase 2.
"""

from __future__ import annotations

import contextvars
import time
import uuid
from collections.abc import Awaitable, Callable, MutableMapping
from typing import Any

import structlog

from llmops_core.logging import get_logger

REQUEST_ID_HEADER = "x-request-id"
_MAX_REQUEST_ID_LEN = 128

_request_id: contextvars.ContextVar[str | None] = contextvars.ContextVar("request_id", default=None)

Scope = MutableMapping[str, Any]
Receive = Callable[[], Awaitable[MutableMapping[str, Any]]]
Send = Callable[[MutableMapping[str, Any]], Awaitable[None]]
ASGIApp = Callable[[Scope, Receive, Send], Awaitable[None]]


def get_request_id() -> str | None:
    return _request_id.get()


def new_request_id() -> str:
    return uuid.uuid4().hex


def _sanitize(raw: str | None) -> str:
    """Accept a client-supplied id only if it is short and printable; else mint one."""
    if raw and len(raw) <= _MAX_REQUEST_ID_LEN and raw.isprintable():
        return raw
    return new_request_id()


class RequestContextMiddleware:
    """Assigns request_id, binds it to log context, echoes it back, logs one access line."""

    def __init__(self, app: ASGIApp) -> None:
        self.app = app
        self.log = get_logger("http")

    async def __call__(self, scope: Scope, receive: Receive, send: Send) -> None:
        if scope["type"] != "http":
            await self.app(scope, receive, send)
            return

        headers = {k.decode().lower(): v.decode() for k, v in scope.get("headers", [])}
        request_id = _sanitize(headers.get(REQUEST_ID_HEADER))
        token = _request_id.set(request_id)
        structlog.contextvars.clear_contextvars()
        structlog.contextvars.bind_contextvars(
            request_id=request_id,
            method=scope.get("method"),
            path=scope.get("path"),
        )
        started = time.perf_counter()
        status_code = 500

        async def send_wrapper(message: MutableMapping[str, Any]) -> None:
            nonlocal status_code
            if message["type"] == "http.response.start":
                status_code = message["status"]
                raw_headers = list(message.get("headers", []))
                raw_headers.append((REQUEST_ID_HEADER.encode(), request_id.encode()))
                message["headers"] = raw_headers
            await send(message)

        try:
            await self.app(scope, receive, send_wrapper)
        finally:
            elapsed_ms = round((time.perf_counter() - started) * 1000, 2)
            self.log.info("request", status=status_code, latency_ms=elapsed_ms)
            structlog.contextvars.clear_contextvars()
            _request_id.reset(token)
