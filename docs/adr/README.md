# Architecture Decision Records

One file per decision, immutable once accepted (supersede with a new ADR).
Numbering 001–010 follows the master prompt's list; 011+ are additional.

| ADR | Title | Status | Phase |
| --- | --- | --- | --- |
| [001](ADR-001-fastapi.md) | FastAPI for application services | Accepted | 1 |
| 002 | Qdrant as vector database | Planned | 8 |
| [003](ADR-003-hybrid-retrieval.md) | Hybrid retrieval, weighted fusion over RRF | Accepted | 9 |
| [004](ADR-004-reranking.md) | Cross-encoder reranking, available but off by default | Accepted | 9 |
| 005 | Redis for cache / queue / rate-limit state | Planned | 15 |
| 006 | vLLM as inference engine | Planned | 6 |
| 007 | Kong OSS as API Gateway | Planned (pre-decided in architecture.md AD-0.9) | 16 |
| 008 | Kong Upstream as Load Balancer | Planned | 17 |
| 009 | Asynchronous queue for ingestion | Planned | 20 |
| 010 | Kubernetes via minikube | Planned | 32 |
| [011](ADR-011-monorepo-uv-workspace.md) | Monorepo with uv workspace + poethepoet task runner | Accepted | 1 |
| [012](ADR-012-structlog-json-logging.md) | structlog JSON logging with request context | Accepted | 1 |
| [013](ADR-013-health-probe-model.md) | Liveness / readiness / startup probe model | Accepted | 1 |
| [014](ADR-014-a2a-agent-protocol.md) | A2A for agent-to-agent, alongside MCP | Accepted | 26b |
| [015](ADR-015-release-manifest-versioning.md) | Nine versioned artifact classes + release manifest | Accepted | 10 → 35 |
| [016](ADR-016-hitl-risk-tiers.md) | Risk tiers + durable human-in-the-loop interrupt | Accepted | 28b |
| [017](ADR-017-otel-genai-semconv.md) | Adopt OTel GenAI semantic conventions where they exist | Accepted | 2 → 22 |
| [018](ADR-018-provider-abstraction.md) | One provider interface for mock / local / external LLMs | Accepted | 2 |
| [019](ADR-019-authorization-model.md) | RBAC for capabilities, ABAC for data, ACL as a query predicate, ACL-scoped cache | Accepted | 11 → 24 |
| [020](ADR-020-fastembed-over-sentence-transformers.md) | fastembed (ONNX) rather than sentence-transformers | Accepted | 8 |
| [021](ADR-021-prompt-registry-design.md) | Prompt registry: files in Git, hashes for immutability, pointers for stages | Accepted | 10 |
| [022](ADR-022-evaluation-design.md) | Evaluation: generated ground truth, arithmetic metrics, library not service | Accepted | 11 |

Template: [ADR-000-template.md](ADR-000-template.md)
