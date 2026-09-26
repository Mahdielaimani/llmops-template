"""X-Request-ID propagation. Pure ASGI, not BaseHTTPMiddleware, which buffers
the response body and would break token streaming."""

from __future__ import annotations

import contextvars
import time
import uuid
from typing import Any

import structlog
from starlette.types import ASGIApp, Receive, Scope, Send

from llmops_core.logging import get_logger

REQUEST_ID_HEADER = "x-request-id"
_MAX_REQUEST_ID_LEN = 128

_request_id: contextvars.ContextVar[str | None] = contextvars.ContextVar("request_id", default=None)


def get_request_id() -> str | None:
    return _request_id.get()


def _sanitize(raw: str | None) -> str:
    """Trust a client-supplied id only if short and printable — it reaches the logs."""
    if raw and len(raw) <= _MAX_REQUEST_ID_LEN and raw.isprintable():
        return raw
    return uuid.uuid4().hex


class RequestContextMiddleware:
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
        status_code = 500  # stays 500 if the app dies before sending response.start

        async def send_wrapper(message: Any) -> None:
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
