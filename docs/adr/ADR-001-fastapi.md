# ADR-001: FastAPI for application services

**Status:** Accepted
**Date:** 2026-09-17
**Phase:** 1

## Context
Application services (llm-application, rag-service, agent-service,
model-gateway, embedding-service) are I/O-bound: they await model
servers, vector DB, Redis, Postgres. They must stream tokens (SSE),
expose OpenAPI for the gateway and tests, and stay small enough to run
four replicas in ~1.5 GB RAM. The team language is Python (ML ecosystem).

## Decision
FastAPI on uvicorn, ASGI, async handlers, Pydantic v2 models for
request/response/config. One app factory per service (`create_app`),
shared primitives from `llmops-core`.

## Alternatives considered
| Option | Why not |
|---|---|
| Flask / Django | WSGI, no native async or streaming; would block on every model call |
| Litestar | comparable; smaller ecosystem, fewer examples for vLLM/OpenAI-compat tooling |
| Go (Gin/Fiber) | faster per-request, but the bottleneck is the GPU not the API; splits the codebase from the ML tooling |
| Node/Express | same argument; Python tooling (transformers, qdrant-client, mlflow) drives the choice |

## Trade-offs
+ async streaming, OpenAPI for free, Pydantic validation shared with config
+ tiny footprint per replica (~60 MB RSS idle)
− Python GIL: CPU-heavy work (reranking, BM25 over large corpora) must move
  to workers or separate services — this is deliberate and teaches the split
− uvicorn single-process per container; horizontal scaling is via replicas
  (Phase 18), not worker counts, which matches K8s practice

## Consequences
- `BaseHTTPMiddleware` is avoided (breaks streaming); middleware is pure ASGI.
- Every service gets `/health/live` and `/health/ready` from `llmops-core`.
- Dependencies (DB pools, HTTP clients) are created in `lifespan`, never at import.
