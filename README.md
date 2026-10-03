# LLMOps Platform Lab — Enterprise Financial Document Assistant

**Status: LOCAL / PRODUCTION-LIKE.** Not production-ready. See
[docs/production-readiness.md](docs/production-readiness.md) (Phase 40) for the gate.

A miniature AI platform, built and operated phase by phase like a
production system: gateway → load balancer → stateless app replicas →
RAG/agents → model gateway → vLLM on GPU, with queues, evaluation,
observability, security, CI/CD, Kubernetes, failure engineering.

Guiding rule: **measure → find the bottleneck → optimize → scale the bottleneck → validate → observe → iterate.**

## Current phase

| Phase | Status |
| --- | --- |
| 0 Environment audit + architecture | done — [docs/environment.md](docs/environment.md), [docs/architecture.md](docs/architecture.md), [docs/roadmap.md](docs/roadmap.md) |
| 1 Repository + engineering foundation | done — this README, `llmops-core`, `llm-application`, Compose `core` profile |
| 2 Basic LLM application (`/chat`) | done — provider abstraction, SSE streaming, TTFT/TPOT/ITL |
| 3 Classical ML serving comparison | done — [docs/classical-ml-vs-llm-serving.md](docs/classical-ml-vs-llm-serving.md) |
| 4 Transformer inference concepts | done — [docs/inference.md](docs/inference.md) |
| 5 Prefill / decode / KV cache | done — [prefill-decode.md](docs/prefill-decode.md), [kv-cache.md](docs/kv-cache.md) |
| 6–7 LLM serving + vLLM | **deferred** — needs disk cleanup, [environment.md](docs/environment.md) §7.3 |
| 8 RAG — ingestion + ACL-filtered retrieval | done — [docs/rag.md](docs/rag.md) |
| 9 Hybrid retrieval + reranking | next |

Full plan: [docs/roadmap.md](docs/roadmap.md). Decisions: [docs/adr/](docs/adr/).

Design references:
- [docs/security.md](docs/security.md) — seven enforcement points, 14-threat model, trust boundaries, and technology alternatives with trade-offs
- [docs/rag.md](docs/rag.md) — ingestion pipeline, ACL as a query predicate, measured retrieval quality against planted ground truth
- [docs/kv-cache.md](docs/kv-cache.md) — the concurrency ceiling: 56 KiB/token, 11 sequences at 4k context, and why PagedAttention exists
- [docs/prefill-decode.md](docs/prefill-decode.md) — two phases, two bottlenecks; the KV cache proved exact
- [docs/inference.md](docs/inference.md) — tokenization, embeddings, attention, Q/K/V, logits and sampling, measured from runnable experiments
- [docs/metrics-and-capacity.md](docs/metrics-and-capacity.md) — latency / TTFT / ITL / TPOT / throughput / VRAM / cost formulas, worked examples on this GPU, and the metric+span naming map
- [docs/agent-platform.md](docs/agent-platform.md) — MCP vs A2A, fan-out/join, skills, verification (L1–L4), HITL risk tiers
- [docs/versioning.md](docs/versioning.md) — nine versioned artifact classes and the release manifest
- [docs/business-requirements.md](docs/business-requirements.md) — SLOs, quality gates, capacity assumptions

## Quick start

Prerequisites (audited machine): Python 3.12 via `uv`, Docker Desktop with WSL2 + NVIDIA runtime.

```bash
cp .env.example .env          # optional; defaults work with the mock provider
uv sync                       # creates .venv, installs workspace
uv run poe check              # lint + mypy --strict + unit tests
uv run poe up                 # docker compose --profile core up -d --build
curl -i http://localhost:8080/health/live
curl -s http://localhost:8080/health/ready | jq

# POST /chat — ungrounded chat, mock provider by default
curl -s localhost:8080/chat -H 'content-type: application/json'   -d '{"messages":[{"role":"user","content":"hello"}],"max_tokens":8}' | jq

# streaming (SSE): deltas, then a terminal `done` event carrying the telemetry
curl -N localhost:8080/chat -H 'content-type: application/json'   -d '{"messages":[{"role":"user","content":"hello"}],"stream":true}'

uv run poe down
```

`/chat` returns `answer`, `request_id`, `model`, `finish_reason`, `latency_ms`,
`ttft_ms`, `tpot_ms`, `tokens_per_second`, token counts, and `grounded: false` —
it is ungrounded by contract; `/rag` (Phase 8) is the grounded path. Token counts
and `ttft_ms` are `null` when the provider does not report them rather than
estimated, because a guessed number corrupts the cost model.

Linux/CI: `make check`, `make up` (delegates to the same `poe` tasks).

Run the API on the host without Docker (hot reload, human-readable logs):

```bash
LLMOPS_APP__LOG_FORMAT=console uv run poe dev
```

## Tasks

| Task | What |
| --- | --- |
| `poe lint` / `poe fmt` | ruff check + format |
| `poe typecheck` | mypy strict over `packages/` and `apps/` |
| `poe test-unit` / `poe test-integration` / `poe cov` | pytest |
| `poe check` | lint + typecheck + unit — run before every commit |
| `poe up` / `poe down` / `poe ps` / `poe logs` | Compose `core` profile |
| `poe dev` | uvicorn with reload on :8080 |

## Layout

```text
packages/llmops-core/   shared: config (pydantic-settings), structlog JSON logging,
                        request-id ASGI middleware, health registry, error envelope
apps/llm-application/   stateless app service (Layer 5): /health now, /chat /rag /agent later
apps/ services/         one deployable per directory (uv workspace members)
inference/              vLLM config, benchmarks, educational experiments
prompts/ models/        versioned prompt and model registries
data/ evaluations/      corpora and eval datasets/reports
tests/{unit,integration,security,evaluation,load}
infrastructure/{docker,kong,kubernetes,helm,monitoring}
docs/                   architecture, environment, roadmap, requirements, ADRs
```

## Compose profiles

RAM is the first bottleneck on the lab machine (15 GB). Never start every
profile at once. Budget per profile in [docs/environment.md §6](docs/environment.md).

| Profile | Services | Since |
| --- | --- | --- |
| `core` | api, postgres, redis, qdrant | Phase 1 |
| `classical` | classical-ml (`/predict`, sklearn) | Phase 3 |
| `rag` | rag-service (`/rag`, ACL-filtered retrieval) | Phase 8 |
| `inference` | vllm (GPU) | Phase 6 |
| `eval` | mlflow | Phase 13 |
| `edge` | kong | Phase 16 |
| `scale` | api replicas, workers | Phase 18 |
| `observability` | prometheus, grafana, otel-collector, jaeger | Phase 22 |
| `load` | locust | Phase 30 |

## Configuration

Environment variables, prefix `LLMOPS_`, nested with `__`
(`LLMOPS_LLM__PROVIDER=vllm`). All keys in [.env.example](.env.example).
Secrets are `SecretStr` and are redacted in logs. Provider modes:

- `mock` — deterministic, no network, default for tests/CI (MODE C)
- `vllm` / `ollama` — local GPU/CPU model via OpenAI-compatible API (MODE B)
- `openai` / `anthropic` — external API, requires a key you supply (MODE A)

## Conventions

- Every request carries `X-Request-ID` (generated if absent, echoed back, in every log line).
- Errors: `{"error": {"code", "message", "request_id", "details?"}}` — also for framework 404/422.
- `/health/live` never checks dependencies; `/health/ready` does. See ADR-013.
- No secrets in code, images, or logs. `.env` is gitignored.
- Labels: DEMO / LOCAL / PRODUCTION-LIKE / PRODUCTION-READY; simulations say **THIS IS A SIMPLIFICATION**.
