from __future__ import annotations

import json
import logging

import pytest
import structlog

from llmops_core.logging import configure_logging, get_logger


def _capture_json_line(capsys: pytest.CaptureFixture[str]) -> dict[str, object]:
    out = capsys.readouterr().out.strip().splitlines()
    assert out, "expected at least one log line on stdout"
    return json.loads(out[-1])  # type: ignore[no-any-return]


def test_json_line_has_required_fields(capsys: pytest.CaptureFixture[str]) -> None:
    configure_logging(level="INFO", fmt="json", service="svc-test")
    structlog.contextvars.bind_contextvars(request_id="req-123")
    get_logger("t").info("hello", latency_ms=12.5)
    line = _capture_json_line(capsys)
    assert line["event"] == "hello"
    assert line["level"] == "info"
    assert line["service"] == "svc-test"
    assert line["request_id"] == "req-123"
    assert line["latency_ms"] == 12.5
    assert "timestamp" in line and "version" in line


def test_service_survives_contextvars_clear(capsys: pytest.CaptureFixture[str]) -> None:
    configure_logging(level="INFO", fmt="json", service="svc-test")
    structlog.contextvars.clear_contextvars()  # what the request middleware does per request
    get_logger("t").info("after_clear")
    assert _capture_json_line(capsys)["service"] == "svc-test"


def test_secret_like_keys_are_redacted(capsys: pytest.CaptureFixture[str]) -> None:
    configure_logging(level="INFO", fmt="json", service="svc-test")
    get_logger("t").info("cfg", api_key="sk-live-123", password="pw", hf_token="hf_abc", user="u")
    line = _capture_json_line(capsys)
    assert line["api_key"] == "***REDACTED***"
    assert line["password"] == "***REDACTED***"
    assert line["hf_token"] == "***REDACTED***"
    assert line["user"] == "u"
    raw = capsys.readouterr().out
    assert "sk-live-123" not in raw


def test_stdlib_loggers_share_formatter(capsys: pytest.CaptureFixture[str]) -> None:
    configure_logging(level="INFO", fmt="json", service="svc-test")
    logging.getLogger("third.party").warning("legacy %s", "msg")
    line = _capture_json_line(capsys)
    assert line["event"] == "legacy msg"
    assert line["level"] == "warning"
    assert line["logger"] == "third.party"


def test_httpx_request_logs_are_suppressed(capsys: pytest.CaptureFixture[str]) -> None:
    """qdrant-client issues one httpx INFO line per search, which would bury the
    retrieval line carrying the actual metrics."""
    configure_logging(level="INFO", fmt="json", service="svc-test")
    logging.getLogger("httpx").info("HTTP Request: POST /points/query 200 OK")
    assert capsys.readouterr().out.strip() == ""
    logging.getLogger("httpx").warning("connection pool exhausted")
    assert "connection pool exhausted" in capsys.readouterr().out
