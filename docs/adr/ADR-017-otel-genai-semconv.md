# ADR-017: Adopt OpenTelemetry GenAI semantic conventions where they exist

**Status:** Accepted
**Date:** 2026-09-26
**Phase:** 2 (naming), 22 (exporters)

## Context
`docs/metrics-and-capacity.md` §9 originally invented a full telemetry vocabulary
(`llmops_ttft_seconds`, `llmops_tokens_total{direction}`, …). The
OpenTelemetry GenAI semantic conventions now define names for the common LLM
attributes and two metrics, with a stable core for chat and embeddings and
native support in Datadog (since OTel v1.37) and Grafana. Agent and
tool-orchestration conventions are explicitly still settling.

The cost of choosing is asymmetric. No instrumentation code existed when this
was decided — Phase 2 was writing the first log fields. Renaming later means
rewriting dashboards, alert rules and recorded queries built on the old names.

## Decision
**Where semconv defines a name, use it. Where it does not, use a clearly
custom namespace and say so.**

| Concept | Name used | Source |
|---|---|---|
| Model requested | `gen_ai.request.model` | semconv |
| Finish reason | `gen_ai.response.finish_reason` | semconv |
| Prompt tokens | `gen_ai.usage.input_tokens` | semconv |
| Completion tokens | `gen_ai.usage.output_tokens` | semconv |
| LLM call duration | `gen_ai.client.operation.duration` | semconv (Phase 22) |
| Token usage metric | `gen_ai.client.token.usage` | semconv (Phase 22) |
| Time to first token | `llm.ttft_ms` | **custom — no semconv equivalent** |
| Time per output token | `llm.tpot_ms` | **custom** |
| Inter-token latency | `llm.itl_ms` | **custom** |
| Output token rate | `llm.tokens_per_second` | **custom** |
| Queue wait | `llmops_queue_wait_seconds` | **custom** |
| KV-cache pressure | `vllm:request_num_preemptions` | **upstream vLLM ≥ 0.30** |
| Cost | `llmops_cost_usd_total` | **custom** |
| Prompt / index / release version | `prompt_version`, `index_version`, `release_id` | **custom** (see ADR-015) |

Three consequences worth stating:

- **TTFT is not in semconv.** It is the metric an LLM platform is judged on
  (SLO-2), and there is no standard name for it. Keeping it under `llm.*`
  signals "ours" rather than pretending it is standard.
- **KV-cache pressure is no longer derived.** vLLM v0.30.0 emits
  `vllm:request_num_preemptions`; scraping it beats inferring pressure from
  GPU-utilisation percentage, which `docs/metrics-and-capacity.md` §8 already
  called the misleading signal.
- **Agent telemetry waits.** Phase 27 will not be built on the unsettled agent
  conventions; it uses `llmops_agent_*` until those stabilise, and this ADR is
  revisited then.

## Alternatives considered
| Option | Why not |
|---|---|
| Keep the invented `llmops_*` vocabulary everywhere | no vendor dashboard understands it; every future integration needs a mapping layer; the naming would have to be defended in interviews rather than cited |
| Adopt semconv wholesale, including agents | agent and tool conventions are explicitly still settling; building Phase 27 on them risks a second rename |
| Wait for semconv to cover TTFT before deciding | blocks Phase 2 on an external roadmap; TTFT is needed now and may never be standardised |
| Use a vendor's proprietary LLM schema (Datadog, Langfuse) | couples the platform to one backend, which is the thing OTel exists to prevent |

## Trade-offs
+ dashboards, alerts and traces are portable across backends with no mapping layer
+ a reader who has seen any other OTel-instrumented LLM system recognises the fields
+ semconv-defined names are documentation for free
− two namespaces coexist (`gen_ai.*` and `llm.*` / `llmops_*`), so the boundary
  must be stated — this table is that statement
− semconv is versioned and will move; a future version may define TTFT and force
  a migration of exactly the fields listed as custom
− the stability claims (v1.37 baseline, v1.41 additions) came from secondary
  sources and are **unverified against the specification**

## Consequences
- `apps/llm-application/routes/chat.py` emits `gen_ai.*` from Phase 2 onward;
  there is no legacy name to migrate.
- `docs/metrics-and-capacity.md` §9 is rewritten around this table.
- Phase 22 configures OTel exporters, not a translation layer.
- Phase 6 scrapes vLLM's `/metrics` rather than computing KV pressure.
- Before Phase 22, read the specification directly
  (`https://opentelemetry.io/docs/specs/semconv/gen-ai/`) and correct any name in
  this table that the spec contradicts. The table, not the blog posts it came
  from, is the contract.
