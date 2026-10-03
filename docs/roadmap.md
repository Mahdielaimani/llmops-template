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
| --- | --- | --- | --- | --- |
| 0 | Environment audit + architecture | ☑ | `docs/environment.md`, `docs/architecture.md`, `docs/roadmap.md` | **your confirmation** |
| 1 | Repository + engineering foundation | ☑ | uv workspace, py3.12 pin, config system (pydantic-settings), JSON logging, pytest, task runner, Dockerfile base, `.env.example`, `docs/business-requirements.md` (SLOs, constraints), ADR-001 | tests green, `docker compose --profile core up` healthy |
| 2 | Basic LLM application | ☑ | `POST /chat` with `LLMProvider` abstraction: `mock` + `openai`-compatible client; request_id, timeouts, error model, SSE streaming; response has latency/tokens | curl demo with mock; external key optional |
| 3 | Classical ML serving comparison | ☑ | sklearn model → artifact → FastAPI `/predict`; `docs/classical-ml-vs-llm-serving.md` | side-by-side latency/throughput numbers |

## Block B — Inference internals (GPU)

| # | Phase | Status | Key deliverables | Gate to next |
| --- | --- | --- | --- | --- |
| 4 | Transformer inference concepts | ☑ | `inference/experiments/` with tiny HF model: tokenizer → embeddings → attention → logits → sampling; `docs/inference.md` | notebook-free scripts run on CPU |
| 5 | Prefill / Decode / KV Cache | ☑ | `06_prefill_decode.py` (cache proved exact, 6.5x work saved), `07_kv_cache.py` (capacity plan: 11 seqs @ 4k), `docs/prefill-decode.md`, `docs/kv-cache.md`. **Tables not plots** — matplotlib deferred to Phase 30 (disk at 98%) | numbers cross-check metrics-and-capacity §5 |
| 6 | LLM serving + vLLM | ⊘ | `inference/vllm/` compose service (GPU), Qwen2.5-1.5B then 7B-AWQ, OpenAI-compatible endpoint, startup probe; ADR-006 | **needs: disk cleanup, internet, model download** |
| 7 | Streaming + inference metrics | ⊘ | TTFT / ITL / TPOT / tok/s instrumentation in provider; `benchmarks/` harness | metrics visible in `/chat` response + logs |

## Block C — RAG core

| # | Phase | Status | Key deliverables | Gate to next |
| --- | --- | --- | --- | --- |
| 8 | RAG | ☑ | ingestion pipeline (parse/clean/chunk/metadata/version), Qdrant, embedding-service, `/rag` with citations; `docs/rag.md`; ADR-002 | grounded answer with citations on sample financial docs |
| 9 | Hybrid retrieval + reranking | ☑ | BM25 + RRF fusion + CPU cross-encoder; configurable weights; **chunk sizing must account for 2.34 chars/token on financial text (Phase 4)**; ADR-003, ADR-004 | vector vs hybrid vs hybrid+rerank comparison |
| 10 | Prompt / model versioning | ☐ | `prompts/` registry (immutable released files + semver + stage pointers), `models/registry.yaml`, promote/rollback commands; first 3 of the 9 artifact classes in `docs/versioning.md` | rollback demo; CI fails on edit of a released prompt file |
| 11 | Evaluation | ☐ | `data/evaluation/` dataset, retrieval metrics (Recall@K, MRR, ...), generation metrics, **ACL pre-filter as a query predicate + hard-fail test for Q-6 = 0 (ADR-019)**, `docs/evaluation.md` | baseline report; zero unauthorized retrievals |
| 12 | LLM-as-a-Judge | ☐ | judge prompts, bias/calibration doc, human-vs-judge comparison | **needs: external API key (recommended)** |
| 13 | MLflow experiments | ☐ | `eval` profile, experiments A/B/C tracked | MLflow UI comparison |
| 14 | Regression testing | ☐ | Promptfoo config, baseline compare, deliberate regression detected | CI-style PASS/FAIL |
| 15 | Redis caching | ☐ | exact + embedding cache, TTL, hit/miss metrics; Redis cache vs KV cache doc; **cache key MUST include `acl_scope_hash` — two clearances must never share an entry (ADR-019); measure the hit-rate cost of that fragmentation**; ADR-005 | latency delta measured; cross-clearance cache test passes |

## Block D — Platform / distributed systems

| # | Phase | Status | Key deliverables | Gate to next |
| --- | --- | --- | --- | --- |
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
| --- | --- | --- | --- | --- |
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
| 29 | Security | ☐ | `docs/security.md` **written (Phase 4 interlude)** — 7 enforcement points, 14-threat model, trust boundaries, tool alternatives; remaining work: implement tenant isolation, audit log, dependency + image scan, run the 5 attack simulations incl. cross-tenant cache leak | all 5 detected & mitigated |

## Block F — Scale

| # | Phase | Status | Key deliverables | Gate to next |
| --- | --- | --- | --- | --- |
| 30 | Load testing | ☐ | Locust **as a container on the Compose network** (Phase 3 found ~43 ms of host-side port-proxy overhead), scenarios 1→100 users (500 if RAM allows), graphs, bottleneck report | bottleneck named with evidence |
| 31 | Scaling experiments | ☐ | stage 1–4; API replicas vs inference knobs; `docs/scaling.md` | "scale the bottleneck" proven with numbers |
| 32 | Kubernetes | ☐ | minikube (GPU), manifests: ns/deploy/svc/cm/secret, Kong Ingress Controller (Gateway API), probes incl. startup for vLLM; `docs/kubernetes.md`; ADR-010 | stack runs on minikube |
| 33 | Autoscaling | ☐ | HPA on CPU + custom metric (queue depth / concurrency); GPU capacity not autoscaled — ⊘ documented | HPA scales API pods, p95 unchanged |

## Block G — Delivery / operations

| # | Phase | Status | Key deliverables | Gate to next |
| --- | --- | --- | --- | --- |
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
| --- | --- | --- |
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
- 2026-10-03 — **Re-sequenced.** Phases 6, 7, 32 and 33 deferred pending disk: 29 GB free of 953 GB, and vLLM needs ~10 GB of image plus ~5 GB of weights. `docs/environment.md` §7.3 has the cleanup options; pruning touches the owner's other Docker projects so it is their decision. Owner notified by email. The remaining 32 phases do not need the GPU and proceed in order — deferring is the project's own principle applied to itself.
- 2026-10-03 — Phase 9 complete, and the expected result did not happen. Built BM25 with the same ACL predicate, RRF and weighted fusion, and an ONNX cross-encoder reranker; wrote a 12-question hard set first because Phase 8's Recall@3 of 1.000 left no headroom to measure anything. **Measured: RRF hybrid scored WORSE than dense alone (MRR 0.669 vs 0.778) and the reranker came out exactly equal for 98 ms of extra latency. Weighted fusion won at 0.861.** The per-class breakdown shows the mechanism: BM25 beat dense on bare document codes (1.000 vs 0.750) and scored literally zero on paraphrase, and RRF uses only ranks so it promoted BM25's noise into the fused top-5 — paraphrase MRR fell from 0.778 to 0.178. Weighted fusion survives because normalising collapses a weak retriever's contribution, which is exactly what RRF discards. Default mode changed to hybrid_weighted on that evidence; RRF kept as a mode since it is right at larger candidate pools. Also: my Phase 8 prediction of 200-400 ms for the reranker measured 98 ms, about 3x too pessimistic. ACL holds in all five modes, 0 violations. Honest caveat recorded: n=12 means one question is 0.083 MRR, so only the per-class pattern is trustworthy and Phase 11 needs 100+ questions before these numbers decide anything. `docs/hybrid-retrieval.md`, ADR-003, ADR-004. 108 unit tests.
- 2026-10-03 — Phase 8 complete: `apps/rag-service` with ingestion (validate → parse → clean → metadata → chunk → embed → index) and ACL-filtered retrieval. 12 synthetic financial documents with planted ground truth across all four classifications including a confidential M&A memo. **Measured: Recall@1 0.917, Recall@3 1.000, MRR 0.958, retrieval p50 34 ms. Q-6 hard gate: 0 unauthorized retrievals across 72 queries x 6 identities.** ACL is a query predicate per ADR-019 with an independent re-check costing 0.1 ms; expired auditors retrieve zero candidates (the case RBAC cannot express); all six identities produce distinct cache scope hashes ready for Phase 15. Chunking carries the Phase 4 measurement (2.34 chars/token for financial text). fastembed over sentence-transformers to protect the disk budget (ADR-020). `docs/rag.md`. Also fixed: httpx logged one INFO line per Qdrant search, burying the retrieval metrics line.
- 2026-10-02 — Phase 5 complete: `06_prefill_decode.py` implements generation with and without a KV cache and asserts **identical greedy output** (the cache is exact, not an approximation) — 6.5x less total work, per-step cost flat at 1.01x versus 1.11x growth uncached, and the measured allocation matches the byte formula exactly. `07_kv_cache.py` turns the formula into the capacity plan: 56 KiB/token for Qwen2.5-7B-AWQ, 2.62 GB KV budget, **11 concurrent sequences at 4k context**, independently reproducing metrics-and-capacity §5. Also: fp16 7B does not fit at all (weights alone exceed the card), GQA gives a 7x cache saving, and naive max-context reservation wastes 88% on a typical answer — 11 sequences against 89 paged, which is why PagedAttention exists. `docs/prefill-decode.md`, `docs/kv-cache.md`. 75 unit tests.
- 2026-10-02 — Security design written ahead of Phase 29 after an interview-answer review found three flow errors: ACL was post-filter (leaks through absence, silent top-k collapse), audit/runtime were drawn as pipeline stages rather than cross-cutting bands, and the agent loop was linear so tool observations were never re-sanitized. Added the missing budget governor, **cross-tenant cache leak** (pulled forward to Phase 15 — it bypasses perfect ACL enforcement), supply chain (pickle = RCE), and ABAC-vs-RBAC distinction. `docs/security.md`, `docs/interview/security-layer.md`, ADR-019.
- 2026-10-02 — Phase 4 complete: 5 runnable experiments in `inference/experiments/` (real GPT-2 tokenizer, numpy attention and forward pass) + `docs/inference.md` from measured output. **Finding that changes the capacity model:** financial text tokenizes at 2.34 chars/token against 5.70 for prose, so a fixed context budget retrieves ~2.4x less information from financial documents than the generic estimate assumed — feeds chunk sizing in Phase 9. torch deferred to Phase 6 (disk at 98%). 63 unit tests.
- 2026-10-01 — Phase 3 complete: `classical-ml` service (/predict, /predict/batch, /model), versioned 997-byte artifact with its own metrics and sha256, `benchmarks/compare_serving.py`, `docs/classical-ml-vs-llm-serving.md` written from measured numbers. **Measurement finding:** ~43 ms of every host-side request is Docker Desktop's WSL2 port proxy, identical across endpoints — benchmarks now run inside the Compose network and every handler reports its own `latency_ms`. Phase 30 load tests must follow the same rule. 48 unit tests.
- 2026-09-26 — Phase 2 complete: `POST /chat` with provider abstraction (mock / OpenAI-compatible for vLLM+Ollama+OpenAI), SSE streaming, TTFT/TPOT/ITL measurement, upstream-status mapping, no retries by design. Telemetry adopts OTel GenAI semconv from the first line of instrumentation (ADR-017); provider interface recorded in ADR-018. 33 unit tests. SLO-2/SLO-3 conflict flagged in business-requirements.md pending Phase 30.
- 2026-09-24 — Scope additions after review: A2A (26b), skills registry (27a), fan-out/join (27b), verification (28a), HITL risk tiers (28b); release manifest folded into 35. New docs: `metrics-and-capacity.md` (latency/TTFT/ITL/TPOT/throughput/VRAM/cost formulas + instrumentation map), `agent-platform.md`, `versioning.md`. ADR-014/015/016.
- 2026-09-17 — Phase 1 complete: uv workspace, llmops-core, llm-application, Compose core profile, 20 unit + 5 integration tests, ADR-001/011/012/013, business requirements + SLOs.
- 2026-09-17 — Gateway/LB decision changed: Kong OSS (DB-less) replaces custom FastAPI gateway + nginx LB (AD-0.9).
