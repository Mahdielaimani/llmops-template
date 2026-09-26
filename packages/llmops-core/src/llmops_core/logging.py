"""JSON logging. One shape for the whole process — stdlib loggers from uvicorn
and httpx go through the same formatter, so nothing emits a second format."""

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
    """Last line of defence, not the policy: do not log secrets in the first place."""
    for key in list(event_dict):
        if key.lower() in _REDACT_KEYS or key.lower().endswith(("_key", "_secret", "_token")):
            event_dict[key] = "***REDACTED***"
    return event_dict


def _add_service_fields(_: Any, __: str, event_dict: EventDict) -> EventDict:
    # Not contextvars: the request middleware clears those and would drop these.
    event_dict.setdefault("service", _service_name)
    event_dict.setdefault("version", __version__)
    return event_dict


def configure_logging(*, level: str = "INFO", fmt: str = "json", service: str = "app") -> None:
    """Call once at process start. fmt=console is for humans, json for everything else."""
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

    # Our middleware emits a richer access line than uvicorn's.
    for name in ("uvicorn", "uvicorn.error"):
        logging.getLogger(name).handlers.clear()
        logging.getLogger(name).propagate = True
    logging.getLogger("uvicorn.access").disabled = True


def get_logger(name: str | None = None) -> structlog.stdlib.BoundLogger:
    return structlog.get_logger(name)  # type: ignore[no-any-return]
