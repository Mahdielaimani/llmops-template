# Business Requirements — Enterprise Financial Document Assistant

**Status:** v1 (Phase 1). Numbers marked *assumption* are lab assumptions,
chosen to be realistic for a mid-size company and to be *measurable on
this machine*. They are inputs to capacity planning, not facts.

---

## 1. Business problem

Finance, audit and management staff spend hours searching internal
financial documents (annual reports, quarterly statements, budget memos,
audit findings, policies, contracts). Documents are revised often; old
versions must remain traceable. Some documents are restricted (M&A,
compensation, unreleased results). Wrong or unsourced answers are worse
than no answer: a fabricated number in a board memo is a real incident.

**Goal:** an assistant that answers questions about internal financial
documents with grounded, cited answers, only from documents the asking
user is allowed to see, at predictable latency and cost, with quality
that is measured continuously rather than assumed.

## 2. Users and roles (assumption)

| Role | Count | Typical need | Access |
| --- | --- | --- | --- |
| Analyst | ~150 | "What was Q2 EMEA revenue vs plan?" | public + own-department docs |
| Manager | ~40 | comparisons across quarters, summaries | + department-restricted |
| Finance lead / CFO office | ~10 | sensitive: forecasts, M&A, comp | + confidential |
| Auditor (external, time-boxed) | ~5 | evidence lookups, citations mandatory | scoped doc set only |
| Platform admin | ~3 | operate, evaluate, roll back | no document content by default |

~200 named users, **~50 concurrent at peak** (assumption: quarter close).

## 3. Functional requirements

| ID | Requirement |
| --- | --- |
| FR-1 | Ingest PDF / DOCX / Markdown / CSV financial documents via API; asynchronous with job status. |
| FR-2 | Every document has an id, version, owner, classification (`public`, `internal`, `restricted`, `confidential`), effective date. Re-upload creates a new version; old versions stay queryable by admins. |
| FR-3 | Answer natural-language questions (`/rag`) grounded in retrieved passages; every factual claim carries a citation (`doc_id`, `version`, `chunk_id`, page/section). |
| FR-4 | Refuse or say "not found in the documents I can access" instead of guessing. |
| FR-5 | Plain LLM chat (`/chat`) for non-document tasks (drafting, rewriting), clearly labelled as ungrounded. |
| FR-6 | Agentic mode (`/agent`) for multi-step tasks: retrieve → compute (calculator/SQL over extracted tables) → summarise; every step traced. |
| FR-7 | Retrieval respects user permissions **before** anything reaches the model. |
| FR-8 | Streaming responses (SSE) for chat and RAG. |
| FR-9 | Prompts, models, retrieval parameters are versioned; the active version is visible in every response and log. |
| FR-10 | Offline evaluation suite runnable on demand and in CI; online quality signals collected per request. |
| FR-11 | Admin can roll back prompt, model, or retrieval config independently. |

## 4. Non-functional requirements and SLOs

Local SLOs are what this lab must meet on the audited machine (RTX 4070
8 GB, 15 GB RAM). "Prod target" is what a real deployment would commit to;
kept side by side so the gap is explicit.

| ID | Metric | Local SLO (lab) | Prod target | How measured |
| --- | --- | --- | --- | --- |
| SLO-1 | Availability (`/rag`, `/chat`) | 99.0 % over a load-test window | 99.9 % monthly | Prometheus: `1 - 5xx/total` |
| SLO-2 | RAG **TTFT** p95 (streaming) | ≤ 2.0 s | ≤ 1.0 s | span `llm.first_token` |
| SLO-3 | RAG end-to-end p95, ≤ 300 output tokens | ≤ 12 s at 20 concurrent | ≤ 6 s at 200 concurrent | histogram `http_request_duration` |
| SLO-4 | Chat p95, ≤ 300 output tokens | ≤ 8 s at 20 concurrent | ≤ 4 s | same |
| SLO-5 | Retrieval latency p95 (embed + search + rerank) | ≤ 600 ms | ≤ 300 ms | span `rag.retrieve` |
| SLO-6 | Error rate (5xx, excluding 429/503 shed) | ≤ 1 % | ≤ 0.1 % | counter |
| SLO-7 | Ingestion: 50-page PDF searchable within | ≤ 5 min | ≤ 2 min | job timestamps |
| SLO-8 | Decode throughput (7B INT4) | ≥ 15 tok/s per stream | n/a (hardware) | `llm.tokens_per_second` |

> **Open conflict — SLO-2 and SLO-3 versus the capacity model.**
> [metrics-and-capacity.md](metrics-and-capacity.md) §10 predicts TTFT of
> 1.5–2.5 s and end-to-end 9–12 s *at low load*, which meets or exceeds these
> ceilings before any concurrency is applied. As written, SLO-2 and SLO-3 are
> probably not achievable on an 8 GB GPU with a 7B model and 3 000 tokens of
> context. Resolution is deferred to Phase 30, which measures rather than
> argues; the candidate levers are a smaller model on the SLO path, a tighter
> context budget, or prefix caching (§2 of that document). **Do not treat these
> two numbers as committed until Phase 30 closes this note.**

**Quality objectives** (offline eval set, gate for promotion):

| ID | Metric | Threshold |
| --- | --- | --- |
| Q-1 | Retrieval Recall@5 | ≥ 0.80 |
| Q-2 | Retrieval MRR | ≥ 0.65 |
| Q-3 | Faithfulness (judge + spot-checked by human) | ≥ 0.85 |
| Q-4 | Answer correctness | ≥ 0.75 |
| Q-5 | Citation validity (cited chunk actually supports claim) | ≥ 0.90 |
| Q-6 | Unauthorized-document leakage | **0** — hard fail |
| Q-7 | Prompt-injection test suite pass rate | ≥ 0.95, no data exfiltration |

**Cost objectives** (assumption, tracked from Phase 15):

| Item | Target |
| --- | --- |
| Cost per RAG request, local GPU | report as GPU-seconds × amortised $/h |
| Cost per RAG request, external API | ≤ $0.01 at ≤ 4k input / 300 output tokens |
| Judge cost per eval run | ≤ $1 for the 100-item set |

## 5. Security requirements

| ID | Requirement |
| --- | --- |
| SEC-1 | Authentication on every route except health; JWT with role + department + clearance claims (issued by a stub IdP locally, real IdP in prod). |
| SEC-2 | Authorization = ACL filter applied at retrieval (Qdrant payload filter + BM25 filter) and re-checked before context assembly. The model never sees a disallowed chunk. |
| SEC-3 | Tool calls are authorized by the application per user/role; the model may *request*, never *execute*. |
| SEC-4 | Input guardrails: injection heuristics, PII detection, size limits. Output guardrails: citation validation, sensitive-pattern scan, schema. |
| SEC-5 | Secrets only via environment / secret store; never in code, images, or logs. Logs redact key-like fields. |
| SEC-6 | Audit log: who asked what, which documents were retrieved, which model/prompt version answered, which tools ran. |
| SEC-7 | Tenant/department isolation testable: user A must not retrieve user B's restricted document (regression test, hard fail). |
| SEC-8 | Rate limits per user and per role; concurrency cap protecting the GPU. |
| SEC-9 | Containers run non-root, pinned base images, dependency and image scanning in CI. |

## 6. Constraints

- Single 8 GB GPU → models ≤ 7B INT4; concurrency ceiling set by KV cache.
- 15 GB host RAM → Compose profiles, never the full stack at once.
- Windows host → GPU workloads only inside Docker (WSL2).
- Local-first; no cloud spend without explicit approval.
- No real financial data: a synthetic corpus is generated (Phase 8) with
  deliberately planted facts, versions, and restricted documents so
  evaluation has ground truth.

## 7. Assumptions

- Corpus: ~200 documents, ~10 000 pages, ~60 000 chunks of ~400 tokens (assumption).
- Queries: median 25 tokens; retrieval context ≤ 3 000 tokens; answers ≤ 300 tokens.
- Peak: 50 concurrent users, ~0.5 req/s sustained, bursts to 5 req/s.
- Ingestion: ≤ 20 documents/day, bursts of 100 at quarter close.
- Evaluation set: 100 curated Q/A pairs with expected sources, grown over time.

## 8. Capacity assumptions → what they imply

| Assumption | Implication |
| --- | --- |
| 5 req/s burst × ~6 s each ≈ 30 in flight | GPU handles ~8–12 concurrent decodes at 7B INT4 / 2.5 GB KV → **queue + concurrency cap mandatory** (Phase 21) |
| 3 000-token context per request | KV per request ≈ 3k × 2 × layers × kv_heads × head_dim × 2 B — measured Phase 5; drives `max_num_seqs` |
| 60 000 chunks × 1024-d fp32 | ≈ 250 MB in Qdrant — trivial; retrieval is not the bottleneck, reranking is |
| Reranker 20 candidates × cross-encoder on CPU ≈ 200–400 ms | SLO-5 at risk under load → candidate for Incident 12 and for a GPU or smaller reranker |
| 100 docs at quarter close | ingestion worker pool of 2–4, embedding batched, queue depth alert |

## 9. How requirements shape the architecture

| Requirement | Architectural consequence |
| --- | --- |
| FR-7 / SEC-2 / Q-6 | ACL lives in retrieval payloads and in the query filter; a security boundary *below* the model, not a prompt instruction |
| FR-3 / Q-5 | chunks carry `doc_id/version/chunk_id/page`; context builder keeps a citation map; output guardrail validates it |
| FR-2 | Postgres is the document/version system of record; Qdrant/BM25 are derived indexes, rebuildable |
| FR-1 / SLO-7 | queue + workers, not synchronous upload |
| SLO-2 / FR-8 | streaming end-to-end (vLLM → model-gateway → app → Kong → client) with TTFT measured at each hop |
| SLO-3 + 8 GB VRAM | model gateway with fallback to external API when local queue depth exceeds threshold |
| FR-9 / FR-11 | prompt and model registries in Git + MLflow; version stamped on every response |
| Q-* | evaluation is a CI gate, not a notebook |
| SEC-8 | Kong rate limiting at the edge + app-level token budget |

## 10. Out of scope (v1)

- Multi-language documents.
- Fine-tuning; only prompting, retrieval, and model selection.
- Real IdP integration (stub JWT issuer locally).
- Document editing / write-back.
