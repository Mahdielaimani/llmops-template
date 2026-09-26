"""One error shape for every service, framework errors included:
{"error": {"code", "message", "request_id", "details"}}."""

from __future__ import annotations

from typing import Any

from fastapi import FastAPI, Request
from fastapi.exceptions import RequestValidationError
from fastapi.responses import JSONResponse
from pydantic import BaseModel
from starlette.exceptions import HTTPException as StarletteHTTPException

from llmops_core.context import get_request_id
from llmops_core.logging import get_logger

log = get_logger("errors")


class ErrorBody(BaseModel):
    code: str
    message: str
    request_id: str | None = None
    details: dict[str, Any] | None = None


class ErrorResponse(BaseModel):
    error: ErrorBody


class ApiError(Exception):
    status_code: int = 500
    code: str = "internal_error"

    def __init__(self, message: str, *, details: dict[str, Any] | None = None) -> None:
        super().__init__(message)
        self.message = message
        self.details = details


class NotFoundError(ApiError):
    status_code = 404
    code = "not_found"


class UnauthorizedError(ApiError):
    status_code = 401
    code = "unauthorized"


class ForbiddenError(ApiError):
    status_code = 403
    code = "forbidden"


class RateLimitedError(ApiError):
    status_code = 429
    code = "rate_limited"


class UpstreamTimeoutError(ApiError):
    status_code = 504
    code = "upstream_timeout"


class UpstreamUnavailableError(ApiError):
    status_code = 503
    code = "upstream_unavailable"


def _envelope(code: str, message: str, details: dict[str, Any] | None = None) -> dict[str, Any]:
    return ErrorResponse(
        error=ErrorBody(code=code, message=message, request_id=get_request_id(), details=details)
    ).model_dump(exclude_none=True)


def install_error_handlers(app: FastAPI) -> None:
    @app.exception_handler(ApiError)
    async def _api_error(_: Request, exc: ApiError) -> JSONResponse:
        log.warning("api_error", code=exc.code, status=exc.status_code, message=exc.message)
        return JSONResponse(
            status_code=exc.status_code, content=_envelope(exc.code, exc.message, exc.details)
        )

    @app.exception_handler(StarletteHTTPException)
    async def _http_error(_: Request, exc: StarletteHTTPException) -> JSONResponse:
        code = {404: "not_found", 405: "method_not_allowed", 401: "unauthorized"}.get(
            exc.status_code, "http_error"
        )
        return JSONResponse(
            status_code=exc.status_code,
            content=_envelope(code, str(exc.detail)),
            headers=dict(exc.headers or {}),
        )

    @app.exception_handler(RequestValidationError)
    async def _validation(_: Request, exc: RequestValidationError) -> JSONResponse:
        return JSONResponse(
            status_code=422,
            content=_envelope(
                "validation_error", "Request validation failed", {"errors": exc.errors()}
            ),
        )

    @app.exception_handler(Exception)
    async def _unhandled(_: Request, exc: Exception) -> JSONResponse:
        log.exception("unhandled_error", exc_type=type(exc).__name__)
        return JSONResponse(
            status_code=500, content=_envelope("internal_error", "Internal server error")
        )
