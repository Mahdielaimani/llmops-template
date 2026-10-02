# ADR-012: structlog JSON logging with request context

**Status:** Accepted
**Date:** 2026-09-17
**Phase:** 1

## Context
Phase 29 requires one JSON line per event carrying `request_id`,
`trace_id`, model/prompt versions, token counts and latencies, with
secrets never logged. Third-party libraries (uvicorn, httpx, qdrant)
log via stdlib and must not produce a second log shape.

## Decision
`structlog` configured once per process (`configure_logging`). Processors:
contextvars merge → level → logger name → ISO UTC timestamp → service/version
stamp → secret-key redaction → exception formatting → JSON renderer. The
same processor chain is installed as a stdlib `ProcessorFormatter` so
foreign loggers are rendered identically. The uvicorn access log is
disabled; our ASGI middleware emits the access line with `request_id`,
`status`, `latency_ms`.

Process-level fields (`service`, `version`) are added by a processor, not
contextvars, because the request middleware clears contextvars per request
(bug found and fixed during Phase 1 verification).

## Alternatives considered
| Option | Why not |
| --- | --- |
| stdlib `logging` + custom JSON formatter | no context binding; every call site must pass request_id explicitly |
| `loguru` | pleasant API but weak stdlib interop; contextualisation less explicit |
| `python-json-logger` | formatter only; still no context propagation |
| OpenTelemetry logs SDK as the only path | not yet stable everywhere; added in Phase 22 as an exporter, not a replacement |

## Trade-offs
+ every line is greppable/queryable; `request_id` joins logs to traces
+ redaction processor is a safety net (not a substitute for not logging secrets)
− JSON is noisy for humans; `LLMOPS_APP__LOG_FORMAT=console` for local dev

## Consequences
- Never log a `Settings` object, headers dict, or raw request body.
- Phase 22 adds `trace_id`/`span_id` to the same contextvars — no call-site changes.
