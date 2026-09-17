"""Structured JSON logging with request-scoped context.

One line per event, machine-parseable, always carrying ``request_id`` (and
later ``trace_id``) when set via :mod:`llmops_core.context`. Stdlib loggers
from third-party libraries (uvicorn, httpx) are routed through the same
formatter so the whole process emits one log shape.

Secrets policy: never pass raw settings objects or headers to the logger;
``SecretStr`` fields render as ``**********`` by design.
"""

from __future__ import annotations

import logging
import sys
from typing import Any

import structlog
from structlog.types import EventDict, Processor

from llmops_core.version import __version__

_REDACT_KEYS = frozenset({"api_key", "authorization", "password", "token", "secret", "cookie"})
_service_name = "app"


def _redact_secrets(_: Any, __: str, event_dict: EventDict) -> EventDict:
    """Defensive belt-and-braces: mask obviously sensitive keys even if a caller slips."""
    for key in list(event_dict):
        if key.lower() in _REDACT_KEYS or key.lower().endswith(("_key", "_secret", "_token")):
            event_dict[key] = "***REDACTED***"
    return event_dict


def _add_service_fields(_: Any, __: str, event_dict: EventDict) -> EventDict:
    # Process-level fields live here, not in contextvars: request middleware clears
    # contextvars per request and must not be able to drop them.
    event_dict.setdefault("service", _service_name)
    event_dict.setdefault("version", __version__)
    return event_dict


def configure_logging(*, level: str = "INFO", fmt: str = "json", service: str = "app") -> None:
    """Configure structlog + stdlib logging once at process start.

    Args:
        level: root log level name.
        fmt: ``json`` for machines (default, containers) or ``console`` for humans.
        service: service name stamped on every event.
    """
    global _service_name
    _service_name = service

    shared: list[Processor] = [
        structlog.contextvars.merge_contextvars,
        structlog.stdlib.add_log_level,
        structlog.stdlib.add_logger_name,
        structlog.processors.TimeStamper(fmt="iso", utc=True),
        _add_service_fields,
        _redact_secrets,
        structlog.processors.StackInfoRenderer(),
        structlog.processors.format_exc_info,
    ]

    renderer: Processor = (
        structlog.processors.JSONRenderer()
        if fmt == "json"
        else structlog.dev.ConsoleRenderer(colors=sys.stderr.isatty())
    )

    structlog.configure(
        processors=[*shared, structlog.stdlib.ProcessorFormatter.wrap_for_formatter],
        logger_factory=structlog.stdlib.LoggerFactory(),
        wrapper_class=structlog.stdlib.BoundLogger,
        cache_logger_on_first_use=True,
    )

    formatter = structlog.stdlib.ProcessorFormatter(
        foreign_pre_chain=shared,
        processors=[structlog.stdlib.ProcessorFormatter.remove_processors_meta, renderer],
    )
    handler = logging.StreamHandler(sys.stdout)
    handler.setFormatter(formatter)

    root = logging.getLogger()
    root.handlers.clear()
    root.addHandler(handler)
    root.setLevel(level)

    # Route uvicorn's own loggers through our handler; drop its default access log
    # because our request middleware emits a richer one.
    for name in ("uvicorn", "uvicorn.error"):
        logging.getLogger(name).handlers.clear()
        logging.getLogger(name).propagate = True
    logging.getLogger("uvicorn.access").disabled = True


def get_logger(name: str | None = None) -> structlog.stdlib.BoundLogger:
    return structlog.get_logger(name)  # type: ignore[no-any-return]
