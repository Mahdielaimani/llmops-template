# KV Cache — the Concurrency Ceiling

**Status:** LOCAL / arithmetic. Phase 5. Computed by
`inference/experiments/07_kv_cache.py` over the real geometry of real models,
applied to the 8 GB budget in [environment.md](environment.md). Phase 6 checks
it against `nvidia-smi`.

This document answers one question: **how many requests can this card serve at
once, and why that number and not a bigger one.**

---

## 1. What is cached, and why not Q

Each attention layer projects the hidden state three ways:

| | Meaning | Belongs to | Cacheable |
|---|---|---|---|
| **Q** | what am I looking for | the **current** token | no — there is a new one each step |
| **K** | what can I be found by | **every past** token | **yes** — identical to last step |
| **V** | what I contribute | **every past** token | **yes** |

At decode step `t`, Q is computed for the single new token. K and V for the
`t−1` previous tokens are bit-for-bit what they were at step `t−1`, because the
causal mask guarantees a past token never attends forward, so nothing behind it
can change its projection.

**The causal mask is what makes caching correct**, not just convenient.
Demonstrated in `inference/experiments/03_attention.py` (upper triangle of the
attention weights is exactly zero) and asserted in the test suite.

The cache is **exact**: [prefill-decode.md](prefill-decode.md) §2 generates 80
tokens with and without it and gets byte-identical output. Unlike quantization,
caching costs no quality.

---

## 2. The formula

```text
KV bytes per token per sequence = 2 (K and V) · layers · kv_heads · d_head · bytes_per_element
```

Verified against a real allocation in `prefill-decode.md` §5 — the predicted and
measured byte counts match exactly.

Applied to models this lab actually considers, fp16 cache:

| Model | layers | q/kv heads | GQA | **KiB/token** |
|---|---|---|---|---|
| GPT-2 small (fp16) | 12 | 12 / 12 | 1× | **36** |
| Qwen2.5-1.5B-Instruct (fp16) | 28 | 12 / 2 | 6× | **28** |
| **Qwen2.5-7B-Instruct-AWQ** (int4) | 28 | 28 / 4 | 7× | **56** |
| Llama-3.1-8B-Instruct-AWQ (int4) | 32 | 32 / 8 | 4× | **128** |

### GQA is the lever

Note what is *not* in the formula: **the number of query heads, and the
parameter count**. Only `kv_heads` matters.

Qwen2.5-7B has 28 query heads but 4 KV heads. With plain multi-head attention it
would need **392 KiB/token** instead of 56 — and nothing would fit on this card.
Grouped-query attention exists for exactly this reason, and it is why a 7B model
is serveable on 8 GB while Llama-3.1-8B, with 8 KV heads, needs 128 KiB/token
and is considerably harder.

**Quantizing the weights does not shrink the cache.** Weights and KV are
separate budgets; int4 buys you room *for* a cache, it does not make the cache
smaller. Shrinking the cache needs fewer KV heads, fewer layers, shorter
context, or an fp8 cache.

---

## 3. VRAM is four budgets, not one

```text
VRAM = weights + KV cache + activations + engine overhead
```

RTX 4070 Laptop: 8188 MiB = 8.00 GB. Engine overhead ≈ 0.9 GB (CUDA context,
engine, CUDA graphs), activations ≈ 0.3 GB.

| Model | weights | **KV budget left** | verdict |
|---|---|---|---|
| GPT-2 small (fp16) | 0.27 GB | 6.52 GB | OK |
| Qwen2.5-1.5B (fp16) | 3.39 GB | 3.41 GB | OK |
| **Qwen2.5-7B-AWQ (int4)** | **4.18 GB** | **2.62 GB** | **OK** |
| Llama-3.1-8B-AWQ (int4) | 4.40 GB | 2.40 GB | OK |
| Qwen2.5-7B (fp16) | 16.72 GB | **−9.92 GB** | **NO FIT** |

The last row is the point: at fp16 the weights alone are more than twice the
card. **Quantization is not an optimisation here, it is the entry ticket.**

---

## 4. The concurrency ceiling

Qwen2.5-7B-AWQ: 2.62 GB of KV budget, 56 KiB/token.

```text
max_sequences = KV_budget / (KV_bytes_per_token × context_length)
```

| Context | MB per sequence | **max sequences** |
|---|---|---|
| 1 024 | 59 | **44** |
| 2 048 | 117 | **22** |
| **4 096** | **235** | **11** |
| 8 192 | 470 | **5** |
| 16 384 | 940 | 2 |
| 32 768 | 1 879 | 1 — single request only |

**This table is the platform's capacity plan.** It independently reproduces
`metrics-and-capacity.md` §5 (~11 at 4k, ~5 at 8k) from the model geometry
rather than from a remembered number.

### Context and concurrency are the same knob

| | 1 024 | 2 048 | 4 096 | 8 192 |
|---|---|---|---|---|
| 1 seq | fits | fits | fits | fits |
| 2 seqs | fits | fits | fits | fits |
| 4 seqs | fits | fits | fits | fits |
| 8 seqs | fits | fits | fits | — |
| 16 seqs | fits | fits | — | — |
| 32 seqs | fits | — | — | — |

Linear trade: **halving `max_model_len` doubles `max_num_seqs`.** Those are the
two vLLM flags that matter most on a small card, and they are not independent.

### Against the requirements

`business-requirements.md` §8 assumes 5 req/s bursts at ~6 s each → **~30
requests in flight** (Little's Law). At 4k context this card supports **11**.

The queue and concurrency cap in Phase 21 are therefore **arithmetic, not
architectural taste**. And the reason `metrics-and-capacity.md` flags SLO-2 and
SLO-3 as probably unachievable is visible right here: the gap is 3×, and no
amount of API replicas changes a single row of that table.

---

## 5. Fragmentation, and why PagedAttention exists

A naive engine must reserve `max_model_len` per sequence at admission, because
it cannot know how long the answer will be. With a 4 096 reservation:

| Actual tokens | Used | Wasted | **Waste** |
|---|---|---|---|
| 128 | 7 MB | 228 MB | **97%** |
| 512 | 29 MB | 206 MB | **88%** |
| 1 024 | 59 MB | 176 MB | 75% |
| 2 048 | 117 MB | 117 MB | 50% |
| 4 096 | 235 MB | 0 | 0% |

With a realistic 512-token answer against a 4 096 reservation:

```text
naive pre-allocation  →  11 concurrent sequences
paged, block-level    →  89 concurrent sequences    (8×)
```

**PagedAttention does not change the formula in §2.** Bytes per token are
identical. What it changes is *allocation granularity*: fixed-size blocks
assigned on demand, so a sequence occupies what it uses rather than what it
might use. On these numbers that is 11 versus 89 sequences from the same VRAM.

It also enables **sharing**: a system prompt common to every request can be one
set of blocks referenced by many sequences, with copy-on-write. That is prefix
caching, and it is the main TTFT lever for a RAG system where every request
carries the same instructions.

---

## 6. What happens when the budget is exceeded

Three behaviours, in the order a server should prefer them:

| | Behaviour | Cost |
|---|---|---|
| 1 | **QUEUE** — admission control holds the request | latency rises, nothing fails. **The correct response.** |
| 2 | **PREEMPT** — engine evicts a running sequence's blocks, recomputes or swaps later | that request's latency spikes; others survive |
| 3 | **OOM** — CUDA out of memory | process usually dies, **taking every in-flight request with it** |

OOM recovery is a container restart plus a model reload — 8 s on an H200 after
vLLM v0.30.0, one to three minutes on consumer hardware. So **one
over-admission becomes tens of seconds of total unavailability**, not one failed
request.

### The signal to alert on

`vllm:request_num_preemptions` (vLLM ≥ 0.30.0). A rising preemption count is the
earliest honest warning that the card is over-committed.

**GPU utilisation percentage is the wrong signal.** A GPU reads 99% busy while
merely waiting on memory, so it saturates long before it tells you anything.
Preemptions and queue depth are the real saturation indicators —
`metrics-and-capacity.md` §8, and the reason ADR-017 sources KV pressure from
vLLM's metric rather than deriving it.

### Operational conclusion

Cap concurrency **below** the arithmetic ceiling and alert on preemptions,
because the alternative to shedding one request is losing all of them. Phase 21
builds the cap; Phase 37 Incident 5 reproduces the OOM deliberately to prove the
recovery path.

---

## 7. Cross-checks

Independently derived here, consistent with what was written earlier:

| Claim | This doc | Stated in |
|---|---|---|
| Qwen2.5-7B-AWQ KV/token | 56 KiB | `metrics-and-capacity.md` §5 |
| Max sequences @ 4k | 11 | `metrics-and-capacity.md` §5 (~11) |
| Max sequences @ 8k | 5 | `metrics-and-capacity.md` §5 (~5) |
| GPT-2 small KV/token | 36 KiB | `inference.md` §3 |
| Formula matches a real allocation | exact | `prefill-decode.md` §5 |

---

## 8. Interview answer

> "The KV cache is why decode is viable and also why concurrency is bounded. Q
> belongs to the current token; K and V belong to every past token and are
> bit-identical between steps — the causal mask guarantees a past token never
> attends forward, so nothing can change its projection. That asymmetry is the
> whole justification for caching, and the cache is exact: I verified identical
> greedy output with and without it.
>
> The size is `2 × layers × kv_heads × head_dim × 2 bytes` per token per
> sequence. What's *not* in that formula is the parameter count and the number of
> query heads — only KV heads matter, which is why GQA is the lever. Qwen2.5-7B
> has 28 query heads but 4 KV heads, so 56 KiB per token instead of 392; without
> GQA it wouldn't fit on an 8 GB card at all.
>
> On my card: weights at int4 take 4.2 GB, engine and activations 1.2, leaving
> 2.6 GB of cache. At 4k context that's 235 MB per sequence, so about 11
> concurrent sequences. Halving context doubles that — `max_model_len` and
> `max_num_seqs` are the same knob. My requirements imply around 30 requests in
> flight at burst, so the queue and the concurrency cap are arithmetic, not a
> design preference.
>
> PagedAttention doesn't shrink bytes per token; it changes allocation
> granularity. A naive engine reserves max context per sequence and wastes 88%
> on a typical 512-token answer — on my numbers that's 11 sequences instead of
> 89 from the same VRAM. It also lets a shared system prompt be one set of blocks
> across sequences, which is prefix caching.
>
> And when you exceed the budget, the three outcomes are queue, preempt, or OOM —
> and OOM kills the process with every in-flight request, so I cap below the
> ceiling and alert on `vllm:request_num_preemptions`. GPU utilisation is the
> wrong signal; it reads 99% while just waiting on memory."
