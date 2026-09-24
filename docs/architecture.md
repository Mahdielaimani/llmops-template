# Architecture — Enterprise Financial Document Assistant

**Status label:** PRODUCTION-LIKE (local). Nothing here is production-ready.
**Phase 0 version** — this document evolves; each phase appends a
"Current state" delta and ADRs in `docs/adr/`.

---

## 1. What we are building

An internal assistant that answers questions about versioned, access-
controlled financial documents, with citations, measurable quality,
measurable latency and cost, and full operability (observe, scale,
secure, roll back, recover).

The system is deliberately built as a **miniature AI platform**, not a
RAG app. The application (RAG + agents) is one tenant of a platform that
provides: gateway, load balancing, model gateway, inference serving,
queues, evaluation, observability, and delivery.

Business, functional and non-functional requirements will be written in
`docs/business-requirements.md` during Phase 1. Architecture decisions
that depend on them (SLOs, capacity) are marked *TBD-P1* below.

---

## 2. Target end-state (adapted to this machine)

The master-prompt target has two GPUs and multiple model servers. This
laptop has one 8 GB GPU. Adaptation is explicit; nothing is pretended.

```
                          USERS  (curl / Locust / small web client)
                            |
                            v
                +-----------------------+
                |   EDGE / INGRESS      |   Compose: Kong proxy listener :8080
                |   (TLS optional)      |   K8s: Kong Ingress Controller (Gateway API)
                +-----------+-----------+
                            |
                            v
                +-----------------------+
                |   API GATEWAY         |   Kong OSS, DB-less (`kong.yml`)
                |  authn / RBAC / rate  |   Route+Service+plugins: jwt, acl,
                |  limit / request-id   |   rate-limiting, correlation-id,
                |                       |   request-size-limiting, prometheus,
                |                       |   opentelemetry; routes /chat /rag /agent
                +-----------+-----------+
                            |
                            v
                +-----------------------+
                |   LOAD BALANCER       |   Kong Upstream+Targets (round-robin /
                |  health-aware routing |   least-connections, active+passive
                |                       |   health checks) over N replicas
                +-----+-------+---------+
                      |       |       |
                      v       v       v
                 app-api-1 app-api-2 app-api-3     stateless replicas
                      |       |       |            (no host ports)
                      +-------+-------+
                              |
                    AI APPLICATION LAYER  (`llm-application`)
                              |
                +-------------+-------------+
                |                           |
                v                           v
          RAG SERVICE                 AGENT SERVICE
        (`rag-service`)             (`agent-service`)
                |                           |
                v                           v
         RETRIEVAL LAYER                TOOL LAYER
        +-------+-------+           +-------+--------+
        |               |           |                |
        v               v           v                v
     Qdrant           BM25    TOOL BROKER      MCP server      A2A PEERS
   (dense, ACL       (in-proc   (allowlist,     (controlled    (supervisor <-> research /
    filtered)         index)     authz, risk     resources)     financial / compliance /
                                 tier, HITL)                    writer; task lifecycle,
                                                                Agent Card discovery)
        |               |
        +-------+-------+
                v
             FUSION (RRF / weighted)
                v
             RERANKER (CPU cross-encoder)
                v
          CONTEXT BUILDER (token budget, citations)
                v
        +-------------------+
        |   MODEL GATEWAY   |   `model-gateway`: routing, fallback,
        |                   |   retry/timeout, cost, metrics, versions
        +---+-----------+---+
            |           |
            v           v
      PROVIDER B     PROVIDER A / C
      vLLM (Docker,  external API (key required)
      OpenAI-compat) mock (deterministic)
            |
            v
      INFERENCE ENGINE  vLLM: continuous batching, PagedAttention
            |
            v
      GPU: RTX 4070 Laptop, 8 GB VRAM   (single GPU — "GPU 2" is
            |                            simulated via a second vLLM
            v                            instance only when VRAM allows)
       TOKEN STREAM (SSE)  →  gateway  →  user


ASYNC WORKLOADS
  document upload → api → Redis queue (arq/RQ) → ingest-worker ×N
      → parse → clean → chunk → embed → index (Qdrant + BM25) → Postgres metadata
  batch evaluation → queue → eval-worker → MLflow

STATE (only here — app services are stateless)
  PostgreSQL : users, roles, documents, versions, ACL, audit log, jobs
  Redis      : cache (exact + embedding), rate-limit counters, queue, circuit-breaker state
  Qdrant     : chunk vectors + payload (doc_id, version, acl, chunk_id)
  MinIO/FS   : raw documents (object storage stand-in)
  MLflow     : experiments, model/prompt registry metadata

CROSS-CUTTING
  Observability : OpenTelemetry SDK → OTel collector → Prometheus (metrics),
                  Jaeger (traces), JSON logs → stdout; Langfuse-or-Phoenix for LLM traces
  Security      : JWT authn, RBAC, ACL-filtered retrieval, tool allowlist,
                  input/output guardrails, secrets via env only, audit log
                  6 enforcement points, none of them the model (agent-platform.md §5)
  Agent control : A2A task lifecycle (Postgres-durable), fan-out/join with shared
                  budget, skill registry, verification L1-L4, HITL approval gate
                  for write/irreversible actions, kill switch (agent-platform.md)
  Quantitative  : latency/TTFT/ITL/TPOT/throughput/VRAM/cost formulas and the
                  instrumentation map (metrics-and-capacity.md)
  Evaluation    : offline datasets + pytest-eval + Promptfoo, LLM-judge, regression gate
  Delivery      : Git, GitHub Actions CI, Docker images, Compose → minikube/Helm, rollback
  Governance    : 9 versioned artifact classes + release manifest pinning all of
                  them; independent rollback; ADRs; production-readiness checklist
                  (versioning.md)
```

---

## 3. Layer map (who owns what)

| Layer | Component | Tech (local) | Introduced in phase |
|---|---|---|---|
| 1 Client | curl, Locust, tests | — | 2 |
| 2 Edge/Ingress | proxy listener / ingress | Kong proxy :8080; Kong Ingress Controller on K8s | 16 (Compose), 32 (K8s) |
| 3 API Gateway | Kong Route + Service + plugins (`infrastructure/kong/kong.yml`) | Kong OSS 3.x DB-less | 16 |
| 4 Load Balancer | Kong Upstream + Targets + health checks | Kong OSS (same process, distinct objects) | 17 |
| 3b App policy layer | `apps/api-gateway` — thin: token-budget limits, tenant quotas, tool authz (what Kong OSS lacks) | FastAPI | 16, 21 |
| 5 App services | `apps/llm-application` replicas | FastAPI, stateless | 2 → 18 |
| 6 Orchestration | `apps/rag-service`, `apps/agent-service` (agent loop, supervisor, join) | Python | 8, 24, 27 |
| 6b Agent interop | A2A server+client per agent (Agent Card, task lifecycle), MCP client, tool broker | JSON-RPC/SSE over httpx; Postgres task store | 26, 26b |
| 6c Skills | `skills/*/skill.yaml` — prompt + tool allowlist + limits + eval set | YAML + registry | 27a |
| 6d Verification & HITL | verifier agent, deterministic checks, approval gate, kill switch | Python + Postgres | 28b |
| 7 Retrieval/Data | Qdrant, BM25, Postgres, Redis, workers | qdrant, rank-bm25, psycopg, arq | 8, 9, 20 |
| 8 Model Gateway | `services/model-gateway` | FastAPI + httpx | 15 |
| 9 Inference | vLLM (OpenAI-compatible), mock, external | vLLM in Docker | 6 |
| 10 GPU/CPU | RTX 4070 8 GB / 24 threads | NVIDIA runtime | 6 |
| 11 Observability | OTel, Prometheus, Grafana, Jaeger, LLM tracer | — | 22, 23 |
| 12 Security | authn/authz/RBAC/guardrails | PyJWT, custom | 11, 28, 29 |
| 13 Evaluation | `apps/evaluation-service`, datasets | pytest, Promptfoo, MLflow | 11–14 |
| 14 CI/CD | pipelines, manifests | GitHub Actions, Helm | 34, 35 |

---

## 4. Key architecture decisions (Phase 0 level)

Formal ADRs land in `docs/adr/` as each component is built. These are
the decisions the environment forces *now*.

### AD-0.1 Hybrid execution mode
Providers are pluggable behind one `LLMProvider` interface:
`mock` (default in tests/CI), `vllm` (local GPU), `openai`/`anthropic`
(external, opt-in). Selected via config, never hard-coded.
*Why:* 8 GB VRAM and 15 GB RAM cannot host every experiment; CI must
run without a GPU; judge models benefit from a stronger external model.

### AD-0.2 vLLM runs in Docker, never natively
vLLM has no Windows build. Docker Desktop's WSL2 backend + NVIDIA runtime
is verified working. Cost: model weights pass through the VM's RAM at
load time; startup probes must tolerate 1–3 min model load.

### AD-0.3 Model class ≤ 7B INT4
Weights ≤ ~5 GB leave ≥ 2.5 GB VRAM for KV cache. Default models:
`Qwen/Qwen2.5-1.5B-Instruct` (fast iteration) and
`Qwen/Qwen2.5-7B-Instruct-AWQ` (realism). Ungated → no HF token.

### AD-0.4 Compose profiles instead of one giant stack
RAM (not GPU) is the first bottleneck on this machine. Profiles:
`core`, `inference`, `observability`, `tracing-llm`, `eval`, `scale`,
`load`. The README will state which combos are safe.

### AD-0.5 Gateway on :8080, replicas unpublished
Port 8000 belongs to another project on this machine. Internal
replicas expose no host ports; only the edge does. This is also the
correct production shape (no direct access to internals).

### AD-0.6 Python 3.12 via uv
3.14 is the system default but lacks ML wheels. `uv` pins 3.12, manages
the venv and lockfile. Monorepo uses uv workspaces (one `pyproject.toml`
per app/service, shared lock).

### AD-0.7 Kubernetes via minikube (docker driver, GPU passthrough)
Local-first. The pre-existing `aks-llmops` context is **not** used
unless you explicitly ask; cloud spend and credentials are your domain.

### AD-0.10 A2A alongside MCP (not instead of it)
Decided 2026-09-24. MCP connects an agent to tools; A2A connects an agent
to peer agents that have their own loop, model and context. Both are
carried: MCP in Phase 26, A2A in Phase 26b. Agents publish an Agent Card
generated from the skill registry; task state (`submitted → working →
input-required → completed/failed/canceled`) lives in Postgres so a
supervisor restart or a pending human approval does not lose the run.
*Security:* an A2A peer response is untrusted input — sanitized, framed as
data, never concatenated into a system prompt. Peer authorization (which
agent may call which, with what scopes) is enforced by us, not by the
protocol. Detail: `docs/agent-platform.md`.

### AD-0.11 Nine versioned artifact classes + a release manifest
Decided 2026-09-24. Code, prompt, model, embedding model, retrieval/chunk
config, corpus document, index, eval dataset, agent/skill are each
versioned with their own key and rolled back independently; one release
manifest pins all nine plus a config hash, and its id is stamped on every
response and log line. Embedding-model or chunking changes are *breaking*:
new index version, evaluate, alias flip, keep the old collection for
rollback. `latest` is banned. Detail: `docs/versioning.md`.

### AD-0.12 Verification and human interrupt over trust
Decided 2026-09-24. Agent self-reports are generated text, not evidence.
Four layers: deterministic checks in code (citation contains the claimed
figure, arithmetic recomputed, schema, tool allowlist), an independent
verifier agent that never sees the producer's reasoning, trajectory
evaluation of the process, and signed attestation for audit. Abstention
(`insufficient_evidence`) is a first-class outcome measured as coverage
vs accuracy. Actions are classified by risk tier; `write` and
`irreversible` tiers suspend the run **durably** (state in Postgres, slot
released) and resume only on an approval token. Detail:
`docs/agent-platform.md` §4, §6.

### AD-0.9 Kong OSS (DB-less) as API Gateway and Load Balancer
Decided 2026-09-17 after review. Custom FastAPI gateway dropped.
*Why:* realism — production platforms use Kong/Envoy/APISIX, not a
hand-written Python gateway; Kong plugins cover the Phase 16 checklist
(`jwt`, `key-auth`, `acl`, `rate-limiting`, `request-size-limiting`,
`correlation-id`, `prometheus`, `opentelemetry`, per-Service timeouts);
DB-less mode = one declarative `kong.yml`, no Postgres, ~300 MB RAM;
Kong Ingress Controller carries the same objects into Phase 32.
*Gateway vs LB lesson preserved:* `Route`/`Service`/plugins = gateway
(Layer 3); `Upstream`/`Targets`/health checks/`algorithm` = load balancer
(Layer 4). Same binary, two objects, taught separately; replica-kill demo
shows passive+active health checks evicting a target.
*Boundaries:* Kong stays at the edge. It does **not** replace
`services/model-gateway` (model routing, fallback, cost) — that is built
by hand as a lesson; Kong's `ai-proxy` plugin is shown only as contrast.
What Kong OSS cannot do (token-based rate limiting, tenant token budgets,
tool authorization) lives in a thin app-side policy layer.
*Alternatives rejected:* Envoy (xDS overkill locally), APISIX (etcd →
more RAM), Traefik (weak authz plugins), nginx (LB only, no plugins),
custom FastAPI (unrealistic). Full ADR-007/008 at Phase 16/17.

### AD-0.8 Single GPU: "scale inference" is demonstrated honestly
With one 8 GB card we cannot add a second GPU. Inference scaling is
shown by: (a) changing `max_num_seqs` / `gpu_memory_utilization` /
quantization, (b) a second small-model vLLM instance sharing the card
when VRAM allows, (c) the external-API provider as "elastic capacity".
Every such demo is labelled **THIS IS A SIMPLIFICATION** with the
real-world equivalent described.

---

## 5. Request paths

### 5.1 Synchronous RAG query (target state)

```
client ─► Kong :8080 [Route → plugins: jwt, acl, rate-limiting,
   │        correlation-id, otel] ─► Kong Upstream (LB, health-checked)
   │        ─► app-api-N ─► rag-service
   │ Kong: JWT verify, ACL group check, rate-limit, request_id, trace ctx
   │ App:  token budget / tenant quota / tool authz
   │
   rag-service:
     1. query processing (normalize, optional rewrite)
     2. ACL filter derived from user claims   ◄── security boundary
     3. embed query           (embedding-service, cached in Redis)
     4. dense search Qdrant   (payload filter = ACL)   ┐ parallel
     5. BM25 search           (filtered by same ACL)   ┘
     6. fusion (RRF)  →  rerank (cross-encoder)  →  top-k
     7. context builder (token budget, citation map)
     8. model-gateway.generate(prompt_version, model_version, stream=True)
           → provider (vLLM / external / mock)  → token stream
     9. output guardrails (schema, citation validity, PII)
    10. response {answer, citations, request_id, model, prompt_version,
                  latency, ttft, tokens, cost}
```

Every hop emits an OTel span; the trace shows exactly where time goes
(gateway → retrieval → rerank → prefill → decode).

### 5.2 Asynchronous ingestion

```
POST /documents  ─► gateway ─► app-api ─► enqueue(job) ─► 202 {job_id}
                                             │
                                  Redis queue ▼  (depth = backpressure signal)
                                  ingest-worker: parse → clean → metadata →
                                  chunk → embed (batched) → upsert Qdrant →
                                  rebuild BM25 → Postgres: document version row
                                  retries w/ backoff → dead-letter after N
GET /jobs/{id} ─► status
```

### 5.3 Agentic path

```
/agent ─► agent-service: plan → (tool call?) → authorization check ◄── app decides, not LLM
        → tool exec (retrieval / calc / SQL / MCP) → observe → loop (max_iter, budget)
        → final answer; each step traced; multi-agent supervisor adds delegation depth limit
```

---

## 6. Where the bottleneck will be (prediction, to be measured)

Formulas, worked examples and the full instrumentation map:
**[docs/metrics-and-capacity.md](metrics-and-capacity.md)**. Headlines for
Qwen2.5-7B-AWQ on the 8 GB card: KV cache = 56 KiB per token per sequence
→ ~11 concurrent sequences at 4k context (~5 at 8k); decode is
bandwidth-bound at ~30–40 tok/s single-stream; prefill of a 3k-token RAG
context is ~1.7 s and dominates TTFT; requirements imply ~30 requests in
flight at burst versus ~11 slots → queue and load-shedding are arithmetic
necessities, not design taste.

| Stage | Expected on this machine | Why |
|---|---|---|
| Gateway / LB / API | µs–ms, scales with replicas | stateless, CPU plentiful |
| Embedding (CPU) | 20–80 ms / query | small model, 24 threads |
| Qdrant | 5–20 ms | small corpus |
| BM25 | 5–15 ms | in-process |
| Reranker (CPU cross-encoder) | 100–400 ms for 20 candidates | **first CPU bottleneck** — Incident 12 |
| Prefill (GPU) | 100–500 ms | scales with prompt length |
| Decode (GPU) | 15–40 ms/token on 7B INT4 | **dominant**; memory-bandwidth bound |
| KV cache | ~2.5 GB budget | **concurrency ceiling** — VRAM OOM reproducible |
| RAM (host) | 15 GB | limits how many profiles coexist |

Prediction to verify in Phase 30–31: adding API replicas beyond 2 will
not move p95; `max_num_seqs` and context length will.

---

## 7. Non-goals for the local lab

- Multi-node inference, tensor parallelism (one GPU).
- Real TLS certificates / DNS (self-signed only if ingress TLS is shown).
- Cloud deployment (AKS context exists but unused unless requested).
- Production-grade secret manager (env files + K8s Secrets; Vault
  documented as the real answer).

---

## 8. Repository layout (as it will be created in Phase 1)

Follows the master prompt structure (`apps/`, `services/`, `inference/`,
`data/`, `prompts/`, `evaluations/`, `benchmarks/`, `tests/`,
`infrastructure/`, `scripts/`, `docs/`). Deviations, if any, will be
recorded in ADR-001.
