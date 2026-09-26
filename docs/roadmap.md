# Roadmap

Phases follow the master prompt (`CLAUDE.md` §72). Each phase ends with:
implement → test → run → verify → manual validation steps → concepts →
failure modes → observability → scaling → interview answer → **STOP and
wait for confirmation**.

Legend: ☐ not started · ◐ in progress · ☑ done · ⊘ simulated (hardware
limit, labelled THIS IS A SIMPLIFICATION)

Machine constraints that shape the order (see `docs/environment.md`):
8 GB VRAM, 15 GB RAM, 33 GB disk (cleanup required before Phase 6),
vLLM only in Docker, no `make`, internet currently offline.

---

## Block A — Foundation

| # | Phase | Status | Key deliverables | Gate to next |
|---|---|---|---|---|
| 0 | Environment audit + architecture | ☑ | `docs/environment.md`, `docs/architecture.md`, `docs/roadmap.md` | **your confirmation** |
| 1 | Repository + engineering foundation | ☑ | uv workspace, py3.12 pin, config system (pydantic-settings), JSON logging, pytest, task runner, Dockerfile base, `.env.example`, `docs/business-requirements.md` (SLOs, constraints), ADR-001 | tests green, `docker compose --profile core up` healthy |
| 2 | Basic LLM application | ☑ | `POST /chat` with `LLMProvider` abstraction: `mock` + `openai`-compatible client; request_id, timeouts, error model, SSE streaming; response has latency/tokens | curl demo with mock; external key optional |
| 3 | Classical ML serving comparison | ☐ | sklearn model → artifact → FastAPI `/predict`; `docs/classical-ml-vs-llm-serving.md` | side-by-side latency/throughput numbers |

## Block B — Inference internals (GPU)

| # | Phase | Status | Key deliverables | Gate to next |
|---|---|---|---|---|
| 4 | Transformer inference concepts | ☐ | `inference/experiments/` with tiny HF model: tokenizer → embeddings → attention → logits → sampling; `docs/inference.md` | notebook-free scripts run on CPU |
| 5 | Prefill / Decode / KV Cache | ☐ | measured prefill vs decode, KV memory vs context/concurrency plots; `docs/prefill-decode.md`, `docs/kv-cache.md` | graphs in `benchmarks/` |
| 6 | LLM serving + vLLM | ☐ | `inference/vllm/` compose service (GPU), Qwen2.5-1.5B then 7B-AWQ, OpenAI-compatible endpoint, startup probe; ADR-006 | **needs: disk cleanup, internet, model download** |
| 7 | Streaming + inference metrics | ☐ | TTFT / ITL / TPOT / tok/s instrumentation in provider; `benchmarks/` harness | metrics visible in `/chat` response + logs |

## Block C — RAG core

| # | Phase | Status | Key deliverables | Gate to next |
|---|---|---|---|---|
| 8 | RAG | ☐ | ingestion pipeline (parse/clean/chunk/metadata/version), Qdrant, embedding-service, `/rag` with citations; `docs/rag.md`; ADR-002 | grounded answer with citations on sample financial docs |
| 9 | Hybrid retrieval + reranking | ☐ | BM25 + RRF fusion + CPU cross-encoder; configurable weights; ADR-003, ADR-004 | vector vs hybrid vs hybrid+rerank comparison |
| 10 | Prompt / model versioning | ☐ | `prompts/` registry (immutable released files + semver + stage pointers), `models/registry.yaml`, promote/rollback commands; first 3 of the 9 artifact classes in `docs/versioning.md` | rollback demo; CI fails on edit of a released prompt file |
| 11 | Evaluation | ☐ | `data/evaluation/` dataset, retrieval metrics (Recall@K, MRR, ...), generation metrics, `docs/evaluation.md` | baseline report |
| 12 | LLM-as-a-Judge | ☐ | judge prompts, bias/calibration doc, human-vs-judge comparison | **needs: external API key (recommended)** |
| 13 | MLflow experiments | ☐ | `eval` profile, experiments A/B/C tracked | MLflow UI comparison |
| 14 | Regression testing | ☐ | Promptfoo config, baseline compare, deliberate regression detected | CI-style PASS/FAIL |
| 15 | Redis caching | ☐ | exact + embedding cache, TTL, hit/miss metrics; Redis cache vs KV cache doc; ADR-005 | latency delta measured |

## Block D — Platform / distributed systems

| # | Phase | Status | Key deliverables | Gate to next |
|---|---|---|---|---|
| 16 | API Gateway | ☐ | Kong OSS DB-less (`infrastructure/kong/kong.yml`): jwt, acl (RBAC), rate-limiting, request-size-limiting, correlation-id, prometheus, otel plugins; routes /chat /rag /agent /health; thin app policy layer for token budgets; ADR-007 | internals no longer exposed; Kong Admin API read-only on localhost |
| 17 | Load Balancer | ☐ | Kong Upstream + Targets, round-robin vs least-connections, active+passive health checks, replica-kill demo; Gateway-vs-LB explained on Kong objects; ADR-008 | traffic survives replica failure |
| 18 | Multiple stateless replicas | ☐ | `scale` profile, state moved to Postgres/Redis | 4 replicas, no sticky state |
| 19 | Service discovery | ☐ | Compose DNS names, no hard-coded IPs; K8s Service preview | services resolve by name |
| 20 | Queues + workers | ☐ | Redis queue (arq), ingest-worker, retries, DLQ; ADR-009 | async ingestion with job status |
| 21 | Backpressure + rate limiting | ☐ | concurrency semaphore, token-bucket rate limit, 429/503 behaviour, queue-depth metric | spike demo: 100 req vs 10 slots |
| 22 | Observability | ☐ | OTel SDK, Prometheus, Grafana dashboards (app/LLM/RAG/agent/infra), DCGM or nvidia-smi exporter; Langfuse vs Phoenix decision; `docs/observability.md` | dashboards populated |
| 23 | Distributed tracing | ☐ | end-to-end trace gateway→GPU, latency breakdown | one trace explains p95 |

## Block E — Agents

| # | Phase | Status | Key deliverables | Gate to next |
|---|---|---|---|---|
| 24 | Agent | ☐ | single agent loop, tool registry, max_iter/timeout/cost caps | no infinite loop under adversarial goal |
| 25 | Agentic RAG | ☐ | retrieval-as-tools, self-check, traced steps; `docs/agentic-rag.md` | agent chooses strategy |
| 26 | MCP | ☐ | one MCP server (financial resources/tools), client in agent; authz outside MCP | "MCP ≠ security boundary" demo |
| 26b | **A2A (agent-to-agent)** | ☐ | A2A server+client, Agent Card at `/.well-known/agent-card.json` generated from the skill registry, task lifecycle (submitted→working→input-required→completed/failed/canceled) persisted in Postgres, peer authz + remote-output sanitization; `docs/agent-platform.md` §1; ADR-014 | two agents complete a task across the protocol; supervisor restart mid-task loses nothing |
| 27 | Multi-agent | ☐ | supervisor + research/analysis/security/writer; delegation depth limit; `docs/multi-agent.md` | bounded run with trace |
| 27a | **Skills registry** | ☐ | `skills/*/skill.yaml` (prompt ref, tool allowlist, limits, risk tier, eval set); supervisor routes by skill not by agent name; Agent Card generated from it | adding an agent requires no supervisor change |
| 27b | **Coordination: fan-out / join** | ☐ | parallel branches with shared token+cost budget, per-branch deadline, completion policy (`all`/`first_success`/`quorum`), partial-failure degradation, Redis blackboard | 3 branches join under one budget; one branch killed → answer degrades and says so |
| 28 | Guardrails | ☐ | input/output/tool guardrails, schema validation, citation check | injection payloads blocked |
| 28a | **Verification (trust)** | ☐ | L1 deterministic (cited chunk contains the figure, arithmetic recomputed, schema, allowlist), L2 independent verifier agent, L3 trajectory eval, L4 signed attestation; abstention as a measured outcome (coverage vs accuracy); `docs/agent-platform.md` §4 | fabricated citation is a hard fail before the answer is returned |
| 28b | **HITL interrupt / risk tiers** | ☐ | risk classification per action (read/compute/sensitive-read/write/irreversible), durable suspend to Postgres + A2A `input-required`, single-use resume token bound to approver, auto-deny timeout, kill switch + per-tool circuit breaker; ADR-016 | agent pauses on a write action, releases its slot, survives a restart, resumes on approval |
| 29 | Security | ☐ | threat model, trust boundaries, tenant isolation, audit log, dependency + image scan; 4 attack simulations; `docs/security.md` | all 4 detected & mitigated |

## Block F — Scale

| # | Phase | Status | Key deliverables | Gate to next |
|---|---|---|---|---|
| 30 | Load testing | ☐ | Locust scenarios 1→100 users (500 if RAM allows), graphs, bottleneck report | bottleneck named with evidence |
| 31 | Scaling experiments | ☐ | stage 1–4; API replicas vs inference knobs; `docs/scaling.md` | "scale the bottleneck" proven with numbers |
| 32 | Kubernetes | ☐ | minikube (GPU), manifests: ns/deploy/svc/cm/secret, Kong Ingress Controller (Gateway API), probes incl. startup for vLLM; `docs/kubernetes.md`; ADR-010 | stack runs on minikube |
| 33 | Autoscaling | ☐ | HPA on CPU + custom metric (queue depth / concurrency); GPU capacity not autoscaled — ⊘ documented | HPA scales API pods, p95 unchanged |

## Block G — Delivery / operations

| # | Phase | Status | Key deliverables | Gate to next |
|---|---|---|---|---|
| 34 | CI/CD | ☐ | GitHub Actions: lint → unit → integration → security → eval gate → build → scan; `docs/ci-cd.md` | PR pipeline green |
| 35 | Rollback + release manifest | ☐ | 9 artifact classes, release manifest pinning all of them, independent rollback per class, index alias flip for breaking embedding/chunking changes, `scripts/pin.py`, "reproduce yesterday" replay; `docs/versioning.md`; ADR-015 | replayed release reproduces its recorded quality-gate metrics |
| 36 | Drift | ☐ | simulated query/embedding drift, monitors + alerts | alert fires |
| 37 | Failure engineering | ☐ | 12 incidents (LLM down, Qdrant, Redis, PG, GPU OOM, model load, latency, concurrency, injection, authz, API timeout, reranker) | each: detect→RCA→fix→regression test |
| 38 | Incident response | ☐ | runbooks, `docs/incident-response.md`, `docs/troubleshooting.md`, incident reports | — |
| 39 | Production simulation | ☐ | scripted scenario 20→100→500 users with injected faults; you diagnose | post-mortem written |
| 40 | Final architecture review | ☐ | `docs/production-readiness.md` checklist, cost model, final diagrams, README | — |
| 41 | Interview preparation | ☐ | 37+ Q&A grounded in this lab | — |

---

## Known blockers by phase

| Phase | Blocker | Owner |
|---|---|---|
| 1 | internet for `uv sync` | resolved 2026-09-17 |
| 1 | task runner: resolved → poethepoet + Makefile shim (ADR-011) | done |
| 6 | ~40 GB Docker cleanup; internet for vLLM image + model | you |
| 12 | external API key for judge (optional but recommended) | you |
| 22 | Langfuse v3 memory (~3 GB) — may switch to Phoenix | decision in phase |
| 30–31 | 500+ users: RAM; will simulate beyond capacity ⊘ | hardware |
| 32 | minikube GPU passthrough on Windows/docker driver — verify early | test in phase |
| 33 | GPU autoscaling impossible locally ⊘ | hardware |

---

## Change log

- 2026-09-14 — Phase 0 complete: audit, mode decision (HYBRID), roadmap created.
- 2026-09-26 — Phase 2 complete: `POST /chat` with provider abstraction (mock / OpenAI-compatible for vLLM+Ollama+OpenAI), SSE streaming, TTFT/TPOT/ITL measurement, upstream-status mapping, no retries by design. Telemetry adopts OTel GenAI semconv from the first line of instrumentation (ADR-017); provider interface recorded in ADR-018. 33 unit tests. SLO-2/SLO-3 conflict flagged in business-requirements.md pending Phase 30.
- 2026-09-24 — Scope additions after review: A2A (26b), skills registry (27a), fan-out/join (27b), verification (28a), HITL risk tiers (28b); release manifest folded into 35. New docs: `metrics-and-capacity.md` (latency/TTFT/ITL/TPOT/throughput/VRAM/cost formulas + instrumentation map), `agent-platform.md`, `versioning.md`. ADR-014/015/016.
- 2026-09-17 — Phase 1 complete: uv workspace, llmops-core, llm-application, Compose core profile, 20 unit + 5 integration tests, ADR-001/011/012/013, business requirements + SLOs.
- 2026-09-17 — Gateway/LB decision changed: Kong OSS (DB-less) replaces custom FastAPI gateway + nginx LB (AD-0.9).
