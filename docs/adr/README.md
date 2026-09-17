# Architecture Decision Records

One file per decision, immutable once accepted (supersede with a new ADR).
Numbering 001–010 follows the master prompt's list; 011+ are additional.

| ADR | Title | Status | Phase |
|---|---|---|---|
| [001](ADR-001-fastapi.md) | FastAPI for application services | Accepted | 1 |
| 002 | Qdrant as vector database | Planned | 8 |
| 003 | Hybrid retrieval | Planned | 9 |
| 004 | Reranking | Planned | 9 |
| 005 | Redis for cache / queue / rate-limit state | Planned | 15 |
| 006 | vLLM as inference engine | Planned | 6 |
| 007 | Kong OSS as API Gateway | Planned (pre-decided in architecture.md AD-0.9) | 16 |
| 008 | Kong Upstream as Load Balancer | Planned | 17 |
| 009 | Asynchronous queue for ingestion | Planned | 20 |
| 010 | Kubernetes via minikube | Planned | 32 |
| [011](ADR-011-monorepo-uv-workspace.md) | Monorepo with uv workspace + poethepoet task runner | Accepted | 1 |
| [012](ADR-012-structlog-json-logging.md) | structlog JSON logging with request context | Accepted | 1 |
| [013](ADR-013-health-probe-model.md) | Liveness / readiness / startup probe model | Accepted | 1 |

Template: [ADR-000-template.md](ADR-000-template.md)
