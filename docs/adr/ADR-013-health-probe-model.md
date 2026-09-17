# ADR-013: Liveness / readiness / startup probe model

**Status:** Accepted
**Date:** 2026-09-17
**Phase:** 1

## Context
Kong (Phase 17) and Kubernetes (Phase 32) both make routing/restart
decisions from health endpoints. An LLM server may take minutes to load a
model; a database outage must not cause every API pod to be restarted;
a replica whose Redis connection is broken must be removed from the pool
but left alive.

## Decision
Two endpoints from `llmops_core.health`, per service:
- `GET /health/live` — constant 200 while the process runs. **No dependency checks.**
- `GET /health/ready` — runs registered checks concurrently with per-check
  timeout; 200 if all `ok`, else 503 with per-check detail. Services
  register checks in `create_app` (Postgres ping, Redis ping, Qdrant
  `/healthz`, model-gateway `/health/ready`).
- Kubernetes `startupProbe` targets `/health/ready` with a large
  `failureThreshold` for model-loading services; liveness only starts
  after startup succeeds.

## Alternatives considered
| Option | Why not |
|---|---|
| single `/health` doing everything | a DB blip restarts pods (liveness) instead of draining them (readiness) |
| readiness that checks *downstream* LLM availability for the API tier | would remove all API replicas when the GPU is saturated — the queue/backpressure layer (Phase 21) must handle that, not the probe |
| TCP-only probes | cannot distinguish "listening" from "model loaded" |

## Trade-offs
+ correct semantics for both Kong upstream health checks and K8s probes
+ per-check latency is a free observability signal
− more endpoints to secure (health routes are unauthenticated; they expose no data)

## Consequences
- Readiness checks must be cheap (<50 ms) and time-boxed (default 2 s).
- Phase 17 demo: kill Redis → replica goes 503 on `/ready`, Kong stops routing to it, container stays up.
