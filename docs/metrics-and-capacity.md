# Metrics, Formulas and Capacity Model

**Status:** LOCAL / PRODUCTION-LIKE. Every formula here is a *model*; the
lab's job is to measure the real value and explain the gap. Hardware
numbers are spec-sheet until Phase 5/7 replaces them with measurements.

This is the quantitative backbone of the platform: what each metric means,
how it is computed, where it is instrumented, and what it implies for
capacity on an RTX 4070 Laptop (8 GB) / 15 GB RAM machine.

---

## 0. Notation

| Symbol | Meaning | Unit |
| --- | --- | --- |
| `P` | prompt (input) tokens | tokens |
| `O` | output (generated) tokens | tokens |
| `N` | model parameters | count |
| `b` | bytes per parameter (FP16=2, FP8/INT8=1, INT4=0.5) | B |
| `L` | transformer layers | count |
| `H_kv` | key/value heads (GQA: fewer than query heads) | count |
| `d_h` | head dimension | count |
| `B` | concurrent sequences in a decode batch | count |
| `C` | context length (prompt + generated so far) | tokens |
| `BW` | GPU memory bandwidth | B/s |
| `F` | effective GPU throughput (FP16/INT4 matmul) | FLOP/s |
| `λ` | arrival rate | req/s |
| `W` | mean time in system | s |
| `ρ` | utilization | 0–1 |

RTX 4070 Laptop reference (spec sheet, to be measured):
`BW ≈ 256 GB/s` (128-bit GDDR6 @ 16 Gbps), `F_fp16 ≈ 30–60 TFLOP/s`
effective dense, VRAM = 8188 MiB, compute capability 8.9.

---

## 1. End-to-end latency decomposition

The only honest definition of "latency" is the sum of named spans. Every
term below is a span in the trace (Phase 23), so a p95 regression can
always be attributed.

```text
T_e2e =  T_edge            Kong: authn, rate-limit, routing
       + T_lb              upstream selection (µs)
       + T_app             request parsing, policy, orchestration
       + T_retrieval       (RAG path only — see §2)
       + T_queue           waiting for an inference slot  ← grows under load
       + T_prefill         prompt processing, ends at first token
       + T_decode          O tokens generated autoregressively
       + T_post            guardrails, citation validation, serialization
```

Streaming changes *perception*, not the sum:

```text
TTFT  = T_edge + T_lb + T_app + T_retrieval + T_queue + T_prefill
T_e2e = TTFT + T_decode + T_post
```

**Rule:** TTFT contains everything before the first token, including
retrieval and queueing. A TTFT regression is therefore *not* necessarily
a GPU problem — that is precisely the diagnostic lesson (Phase 23).

### Retrieval sub-decomposition

```text
T_retrieval = T_embed + max(T_vector, T_bm25) + T_fusion + T_rerank + T_ctx
```
(dense and lexical search run concurrently → `max`, not sum).

Expected on this machine (to verify Phase 8–9):

| Span | Estimate | Note |
| --- | --- | --- |
| `T_embed` | 20–80 ms | CPU, small model, cacheable (Phase 15) |
| `T_vector` | 5–20 ms | Qdrant, ~60k chunks — trivial |
| `T_bm25` | 5–15 ms | in-process |
| `T_fusion` | <1 ms | RRF |
| `T_rerank` | 100–400 ms | **CPU cross-encoder, k=20 — first CPU bottleneck** |
| `T_ctx` | <5 ms | token budgeting, citation map |

---

## 2. TTFT and prefill

Prefill processes all `P` tokens in one batched forward pass →
**compute-bound**.

```text
FLOPs_prefill ≈ 2·N·P                     (dense matmuls: 2 FLOP per param per token)
              + 2·L·P²·(H_kv·d_h)         (attention score+value, quadratic in P)

T_prefill ≈ FLOPs_prefill / F
TTFT      = T_prefill + (everything upstream)
```

The `P²` term is why long RAG contexts hurt TTFT superlinearly: doubling
context more than doubles prefill.

Worked example — Qwen2.5-7B INT4, `P = 3000` (RAG context), `F ≈ 40 TFLOP/s`:

```text
2 · 7e9 · 3000            = 4.2e13 FLOP
2 · 28 · 3000² · (4·128)  = 2.6e13 FLOP     (attention term, non-negligible)
T_prefill ≈ 6.8e13 / 4e13 ≈ 1.7 s          ← model; measure in Phase 5
```

Against SLO-2 (TTFT p95 ≤ 2.0 s) this is *tight* — before adding
retrieval (≈0.5 s) and queueing. Consequence: context budget is a
first-class SLO lever, not a quality-only knob. Mitigations: fewer/shorter
chunks, prefix caching (system prompt reused across requests), chunked
prefill.

---

## 3. ITL, TPOT and decode

Decode generates one token per step for the whole batch. Each step reads
the model weights and the KV cache → **memory-bandwidth-bound**, not
compute-bound. This is the single most important asymmetry in LLM serving.

```text
bytes_per_step ≈ N·b                       (weights, read once per step for the whole batch)
               + Σ_seqs 2·L·H_kv·d_h·b_kv·C_seq    (KV read, per sequence)

t_step ≈ bytes_per_step / BW               (lower bound; real engines hit 50–70 % of BW)

TPOT = mean t_step                         time per output token
ITL  = observed gap between streamed tokens (TPOT + transport + scheduler jitter)
T_decode = O · TPOT
```

Distinction that matters in interviews: **TPOT** is the engine's
generation cost per token; **ITL** is what the *client* observes — it
includes SSE transport, proxy buffering and batching jitter. Kong or a
buffering proxy can inflate ITL while TPOT is unchanged.

Worked example — 7B INT4 (`N·b ≈ 3.5 GB`), single stream, short context:

```text
t_step ≈ 3.5e9 / 2.56e11  ≈ 13.7 ms   → ~73 tok/s theoretical ceiling
at 55 % bandwidth efficiency          → ~40 tok/s realistic single stream
```
SLO-8 (≥ 15 tok/s per stream) therefore has headroom at low concurrency
and is expected to break under batching pressure — exactly the experiment
in Phase 31.

**Why batching is free-ish:** weights are read once per step regardless of
`B`, so `B` sequences cost barely more than one until KV reads dominate.
Aggregate token throughput scales nearly linearly with `B` while per-stream
TPOT degrades slowly — until VRAM runs out.

---

## 4. Throughput

```text
Request throughput     X_req   = B_eff / T_e2e            (Little's Law: L = λ·W)
Output token throughput X_tok  = B_eff / TPOT             (tokens/s, all streams)
Goodput                        = successful requests / s   (excludes 429/503/timeouts)
```

Little's Law is the capacity planner: concurrency `L = λ · W`. From
`docs/business-requirements.md`: burst `λ = 5 req/s`, `W ≈ 6 s` →
`L ≈ 30` in flight. Section 5 shows the GPU supports ~11 — hence the
queue + concurrency cap in Phase 21 is arithmetic, not taste.

Utilization and the latency wall (M/M/1 intuition):

```text
ρ = λ / μ                (μ = service rate)
W = W_service / (1 - ρ)  → at ρ=0.8, 5× the service time; at ρ=0.95, 20×
```
This is why "GPU at 99 %" and "p95 exploded" are the same observation.

---

## 5. VRAM budget — the concurrency ceiling

```text
VRAM = W_model + KV_total + Activations + Overhead

W_model   = N · b · (1 + q)        q ≈ 0.05–0.15 for quant scales/zeros
KV_token  = 2 · L · H_kv · d_h · b_kv          bytes per token per sequence
KV_total  = KV_token · Σ_seqs C_seq
Activations ≈ batch · hidden · L · small       (transient, engine-dependent)
Overhead  ≈ 0.6–1.2 GB                          CUDA context, engine, graphs
```

Worked example — **Qwen2.5-7B-Instruct-AWQ** (L=28, H_kv=4, d_h=128, KV in FP16):

```text
KV_token = 2 · 28 · 4 · 128 · 2 B = 57 344 B ≈ 56 KiB / token / sequence
```

| Item | Size |
| --- | --- |
| Weights INT4 (+10 % quant overhead) | ≈ 4.0 GB |
| CUDA/engine overhead | ≈ 0.9 GB |
| Activations | ≈ 0.3 GB |
| **Left for KV cache** | **≈ 2.8 GB** |

```text
max_concurrent_seqs = KV_budget / (KV_token · C)
  at C = 4096:  2.8e9 / (57344 · 4096) ≈ 11.9  → ~11 sequences
  at C = 8192:  ≈ 5.9                          → ~5 sequences
  at C = 2048:  ≈ 23.8                         → ~23 sequences
```

**Context length and concurrency trade linearly against each other.** This
single table drives `--max-model-len` and `--max-num-seqs` in Phase 6 and
is the mechanism behind Incident 5 (GPU OOM).

Same model at FP16 would need ≈ 14 GB of weights alone → does not fit;
quantization is what makes 7B possible on 8 GB, at a measured quality cost
(Phase 13 experiment).

**PagedAttention** does not reduce `KV_token`; it removes *fragmentation*
and enables sharing, so the practical ceiling approaches the arithmetic
one instead of ~60 % of it.

---

## 6. Token accounting

```text
# Measured, Phase 4: financial text runs 2.34 chars/token against 5.70 for prose,
# so a 400-token chunk of financial text holds ~940 characters where prose holds
# ~2280. Token budgets derived from character counts are wrong by ~2.4x here.
# See docs/inference.md §1.
P = tok(system_prompt) + tok(prompt_template) + tok(history)
  + Σ_{i=1..k} tok(chunk_i) + tok(user_query)

O = generated tokens (bounded by max_tokens)

tokens_request = P + O
```

Agentic amplification — the number that surprises people:

```text
tokens_agent = Σ_{iterations} (P_i + O_i)
P_i grows with accumulated observations (tool outputs are re-sent each step)

with R retries and F parallel branches (fan-out):
tokens_total ≈ F · R · Σ_i (P_i + O_i)

Amplification factor A = tokens_total / tokens_single_shot     typically 3–15×
```

Consequences already designed for: `max_iterations`, `max_delegation_depth`,
per-run token budget, and the retry warning in `CLAUDE.md` §55 — a naive
retry triples GPU load, not just request count.

Cache effect on effective latency and tokens:

```text
T_eff = h · T_cache + (1 - h) · T_full        h = hit rate
tokens_billed_eff = (1 - h) · tokens_request  (exact-match cache)
```

---

## 7. Cost model

### Local GPU
```text
$/hour_amortized = hardware_cost / (lifetime_hours · utilization) + power_kWh · $/kWh
$/request        = GPU_seconds_request · $/hour_amortized / 3600
GPU_seconds_request ≈ T_prefill + O · TPOT     (GPU-exclusive share of the request)
$/1M_tokens      = $/hour_amortized / (3600 · X_tok) · 1e6
```
Reported honestly as GPU-seconds plus the amortization assumption, because
"free because it is my laptop" is not a cost model.

### External API
```text
$/request   = (P/1000)·price_in_per_1k + (O/1000)·price_out_per_1k
$/1M_tokens = published price
```

### Break-even
```text
local_fixed_per_hour = $/hour_amortized
external_variable    = X_req · 3600 · $/request_external

break-even utilization: local becomes cheaper when
  X_req · 3600 · $/request_external  >  $/hour_amortized
```
Low traffic → external API wins. Sustained traffic → local GPU wins. This
comparison is run with real numbers in Phase 31 and drives the
model-gateway fallback policy (Phase 15).

### Cost drivers (elasticity)
| Change | Effect on cost |
| --- | --- |
| context +2× | prefill FLOPs > 2× (P² term); KV memory 2× → concurrency ÷2 → $/req up |
| output +2× | decode time 2× → $/req ≈ 2× (dominant term) |
| concurrency +2× | throughput up, $/req down — **until KV budget is hit** |
| agent iterations +1 | +1 full (P+O) cycle, with a larger P each time |
| quantization FP16→INT4 | weights ÷4, bandwidth ÷4 → ~2–3× throughput; quality delta measured |

---

## 8. Resource metrics beyond the GPU

```text
CPU:   utilization per service; reranker and embedding are CPU-bound here
RAM:   RSS per container vs Compose limit; Docker VM ceiling ≈ 11.7 GiB
VRAM:  used / total, KV-cache utilization %, preemption / swap count
Disk:  model cache size, Qdrant storage, log volume
Net:   inter-service bytes (matters for SSE fan-out)
Queue: depth, oldest-item age, rejection rate
Pool:  DB/HTTP connections in use vs max
```

KV-cache utilization is the LLM-specific saturation signal — more
predictive of latency collapse than GPU-util %, because a GPU can read
99 % busy while merely waiting on memory.

---

## 8b. Measuring on this machine — the port-proxy confound

Measured in Phase 3: **~43 ms of every request issued from the Windows host is
Docker Desktop's WSL2 port proxy.** It is constant and identical for every
endpoint, so it silently swamps any latency conclusion drawn from
`localhost:<port>`.

| Measured from | classical `/predict` p50 |
| --- | --- |
| Windows host | 43.99 ms |
| Compose network | 1.26 ms |
| Inside the handler | 0.26 ms |

Rules that follow, applied from here on:
- benchmarks and load generators run **as containers on the Compose network**
- every handler reports its own `latency_ms` / `total_ms`, so transport is separable
- a p95 regression is attributed from spans (§9), never from a host-side curl

Detail: [classical-ml-vs-llm-serving.md](classical-ml-vs-llm-serving.md) §1.

## 9. Instrumentation map

Every formula term above is one span or one metric. Naming is fixed now so
dashboards and traces line up later.

### Spans (OTel, Phase 23)
```text
http.server.request
 ├─ app.policy
 ├─ rag.retrieve
 │   ├─ rag.embed        attrs: model, cache_hit
 │   ├─ rag.search.vector attrs: top_k, collection, index_version
 │   ├─ rag.search.bm25
 │   ├─ rag.fusion
 │   └─ rag.rerank       attrs: candidates, model
 ├─ llm.generate         attrs: model, model_version, prompt_version, P, O
 │   ├─ llm.queue        ← T_queue
 │   ├─ llm.prefill      ← ends at first token → TTFT marker
 │   └─ llm.decode       attrs: tpot_ms, tokens_per_s
 └─ guard.output
```

### Metrics and attributes (Phase 22)

Names follow the OpenTelemetry GenAI semantic conventions where those define
one, and a custom namespace where they do not. The split is fixed by
**[ADR-017](adr/ADR-017-otel-genai-semconv.md)**; that ADR's table is the
contract, not this summary.

| Concept | Name | Type | Source |
| --- | --- | --- | --- |
| Model requested | `gen_ai.request.model` | attribute | semconv |
| Finish reason | `gen_ai.response.finish_reason` | attribute | semconv |
| Prompt / completion tokens | `gen_ai.usage.input_tokens` / `.output_tokens` | attribute | semconv |
| LLM call duration | `gen_ai.client.operation.duration` | histogram | semconv |
| Token usage | `gen_ai.client.token.usage` | histogram | semconv |
| **Time to first token** | `llm.ttft_ms` | histogram | **custom — no semconv name** |
| Time per output token | `llm.tpot_ms` | histogram | custom |
| Inter-token latency | `llm.itl_ms` | histogram | custom |
| Output token rate | `llm.tokens_per_second` | gauge | custom |
| HTTP request duration | `llmops_request_duration_seconds` | histogram | custom (route-level, not LLM-level) |
| In-flight requests | `llmops_inflight_requests` | gauge | custom |
| Queue depth / wait | `llmops_queue_depth` / `llmops_queue_wait_seconds` | gauge / histogram | custom |
| **KV-cache pressure** | `vllm:request_num_preemptions` | histogram | **upstream vLLM ≥ 0.30.0** |
| vLLM iteration tokens | `vllm:iteration_tokens_total` | counter | upstream |
| GPU VRAM / utilisation | `llmops_gpu_vram_bytes` / `llmops_gpu_utilization` | gauge | custom (exporter) |
| Retrieval stage latency | `llmops_retrieval_duration_seconds{stage}` | histogram | custom |
| Agent iterations / tool calls | `llmops_agent_iterations` / `llmops_tool_calls_total` | histogram / counter | custom — semconv agent conventions still settling |
| Cache hits | `llmops_cache_hits_total{cache}` | counter | custom |
| Cost | `llmops_cost_usd_total{provider}` | counter | custom |

Version labels on every series: `model_version`, `prompt_version`,
`index_version`, `release_id` (see [versioning.md](versioning.md)).

**KV-cache pressure is measured, not derived.** vLLM v0.30.0 (2026-09-22) emits
`vllm:request_num_preemptions`; §8 called GPU-utilisation percentage the
misleading signal, and a preemption count is the honest one.

**Cardinality rule:** never label by `request_id`, `user_id`, or raw query.
Those belong in traces and logs, not in metric labels.

### Log fields (Phase 29)
`timestamp, request_id, trace_id, route, model, model_version,
prompt_version, index_version, release_manifest, P, O, ttft_ms, tpot_ms,
queue_ms, retrieval_ms, latency_ms, cache_hit, cost_usd, status, error`

---

## 10. Capacity summary for this machine

Putting §2–§5 together, the predicted operating envelope (7B INT4, 3k
context, 300 output tokens) — **to be confirmed or refuted in Phase 30–31:**

| Quantity | Model prediction |
| --- | --- |
| KV per token per sequence | 56 KiB |
| Max concurrent sequences @ 4k ctx | ~11 |
| Single-stream decode | ~30–40 tok/s |
| Aggregate decode @ B=8 | ~100–160 tok/s |
| TTFT at low load | ~1.5–2.5 s (prefill-dominated) |
| T_e2e, 300 out tokens, low load | ~9–12 s |
| Sustainable λ before queue growth | ~1.0–1.5 req/s |
| λ from requirements (burst) | 5 req/s → **queue + shed mandatory** |
| First non-GPU bottleneck | CPU reranker (~200–400 ms, serialized per request) |

If the measurements contradict these numbers, the numbers are wrong and
the measurement stands — each phase records the delta and the reason.
