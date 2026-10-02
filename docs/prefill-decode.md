# Prefill and Decode — Two Phases, Two Bottlenecks

**Status:** LOCAL / educational. Phase 5. Numbers from
`inference/experiments/06_prefill_decode.py`, numpy on CPU.

**THIS IS A SIMPLIFICATION.** 2 transformer blocks instead of 12, random
weights, float64, no fused kernels, no batching scheduler, no GPU. The
*relationships* are real — the ratio between the two phases, the shape of the
cost curves, the exactness of the cache. The absolute milliseconds are not; Phase
6 measures vLLM on the RTX 4070.

---

## 1. The two phases

A single generation request is two completely different workloads wearing one
HTTP request.

```text
PROMPT (P tokens)
   │
   ▼
┌──────────────────────────────────────────────┐
│ PREFILL — one forward pass over all P tokens │  ends at first token → TTFT
│ compute-bound: large batched matmuls         │
│ fills the KV cache for positions 0..P-1      │
└──────────────────────────────────────────────┘
   │
   ▼  token 1
┌──────────────────────────────────────────────┐
│ DECODE — N passes over ONE token each        │  per-token gap → ITL / TPOT
│ memory-bound: reads all weights per token    │
│ appends one position to the KV cache          │
└──────────────────────────────────────────────┘
   │
   ▼  tokens 2..N
```

| | Prefill | Decode |
|---|---|---|
| Work shape | one pass over `P` tokens | `N` passes over 1 token |
| Bound by | **compute** (FLOPs) | **memory bandwidth** |
| Parallelism | all `P` positions at once | strictly sequential |
| Scales with | `P` (plus a `P²` attention term) | `N`, output length |
| Metric | **TTFT** | **ITL** observed, **TPOT** internal |
| Batching helps | moderately | **enormously** — weights read once per step for the whole batch |
| Grows the KV cache | by `P` entries | by 1 entry per step |

The operational consequence: **a prompt-heavy workload and an
answer-heavy workload stress different hardware limits and need different
tuning**, even though both arrive as `POST /chat`.

---

## 2. Measured: the cache is exact, not approximate

The experiment generates 80 tokens twice from the same prompt — once
recomputing the whole sequence at every step, once reusing cached K and V.

```text
no cache    [23043, 34071, 45759, 45759, 45759, 45759, 11243, 50214] ...
with cache  [23043, 34071, 45759, 45759, 45759, 45759, 11243, 50214] ...
IDENTICAL   True
```

Greedy decoding is deterministic, so **identical output is a proof**: the KV
cache changes the amount of work, not the result. If that line ever prints
`False`, the causal mask or the position offset is wrong — it is the single most
useful assertion in the whole experiment, and it is in the test suite.

---

## 3. Measured: cost

`P = 19` prompt tokens, 80 generated, 2 blocks, after BLAS warm-up.

| | total | per token |
|---|---|---|
| No cache (recompute everything) | 8 941.7 ms | 111.77 ms |
| Prefill + cached decode | **1 370.0 ms** | **17.13 ms** |
| **Speedup** | **6.5×** | |

Broken into the two phases:

| | Value | What it is |
|---|---|---|
| Prefill | **52.2 ms** | **TTFT** — the user's first visible token |
| Decode, mean step | **16.68 ms** | **TPOT** / ITL |
| Decode, first → last step | 19.61 → 15.94 ms | flat, by design |
| Output throughput | **59.9 tok/s** | `1000 / TPOT` |

Note the ratio: prefill is ~3× one decode step here, for a 19-token prompt. With
a 3 000-token RAG context that flips entirely — prefill dominates TTFT and TTFT
dominates the SLO. That inversion is why `docs/metrics-and-capacity.md` §2 treats
context length as a latency lever, not only a quality one.

---

## 4. Measured: why one curve grows and the other does not

Per-step cost, first quarter versus last quarter of generation, while the
sequence grows from 19 to 99 tokens (5.2×):

| | first | last | growth |
|---|---|---|---|
| No cache | 105.85 ms | 117.69 ms | **1.11×** |
| Cached | 16.63 ms | 16.73 ms | **1.01×** |

The cached curve is **flat**. The uncached one grows.

Why: without a cache, step `t` re-projects K and V for all `t` tokens, so
per-step cost is linear in `t` and the sequence total is **O(n²)**. With a cache,
step `t` projects exactly one token — **O(1) per step, O(n) overall**.

**Honest reading of the 1.11×.** Theory predicts the uncached per-step cost to
grow ~5× over this span, not 1.11×. It does not, because at `d_model = 768` and
`t < 100` the per-step time is dominated by fixed matmul overhead rather than by
the sequence-length-dependent terms. The growth is real and in the right
direction; its magnitude is suppressed by the toy's scale. The earlier version of
this experiment ran 24 tokens without warm-up and showed the uncached version
getting *faster* (0.77×) — the trend only becomes visible once the sequence
length actually spans a wide range and BLAS is warm.

The cached curve is not *perfectly* flat either, and that is correct: attention
still reads a cache that keeps growing, so there is a small linear term that
never disappears. **This is the term that becomes the whole story on a GPU**,
where reading an ever-larger KV cache from HBM is what makes decode
memory-bound.

---

## 5. Measured: the cache's memory cost

```text
measured     2,433,024 B for 99 tokens x 2 layers
formula      2,433,024 B   (2 * 2 * 12 * 64 * 8 B float64)
match        True
```

The formula from `docs/inference.md` §3 predicts the allocation exactly:

```text
KV bytes = 2 (K and V) · layers · kv_heads · d_head · bytes_per_element · tokens
```

numpy is float64 and this toy has 2 layers. Real serving is fp16 with 12–32
layers, which is how 36 KiB/token (GPT-2) and 56 KiB/token (Qwen2.5-7B with GQA)
arise. Turning that into a VRAM budget is [kv-cache.md](kv-cache.md).

---

## 6. What this means for the platform

| Observation | Consequence |
|---|---|
| Prefill is compute-bound, one pass | chunked prefill and prefix caching are the TTFT levers |
| Decode is memory-bound, sequential | more replicas do not help; batching and quantization do |
| TTFT and total latency are different SLOs | SLO-2 and SLO-3 are measured separately (`business-requirements.md` §4) |
| TPOT is flat per sequence | tokens/s per stream is stable; aggregate throughput comes from batch size |
| Cache grows by 1 entry per decode step | memory, not compute, sets the concurrency ceiling |
| The cache is exact | caching is never a quality trade-off — unlike quantization |
| Prefill computes logits for every position, decode needs one | prefill discards `P−1` rows; training needs them, inference does not |

### Where the time goes, for a realistic RAG request

With `P ≈ 3 000` (measured in `docs/inference.md` §1: five financial chunks) and
300 output tokens, on the 8 GB card (predictions from
`metrics-and-capacity.md` §10, to be confirmed in Phase 6):

```text
TTFT  ≈ retrieval (~0.5 s) + queue + prefill (~1.7 s)   → ~2.2 s+
total ≈ TTFT + 300 × TPOT (~25 ms)                      → ~9.7 s
```

Decode is ~75% of total latency and prefill is ~75% of TTFT. Optimising the
wrong one of those two wastes the effort — which is the Phase 0 core principle
applied to a single request.

---

## 7. Verify it yourself

```bash
uv run python inference/experiments/06_prefill_decode.py
uv run poe test-unit -k prefill      # asserts the cache is exact
```

Key assertions in `tests/unit/test_prefill_decode.py`:
- cached and uncached greedy output are **identical**
- cached per-step cost does not grow with sequence length the way uncached does
- measured KV bytes equal the formula exactly
- a wrong position offset breaks output equality (the mask is load-bearing)

---

## 8. Interview answer

> "A generation request is two workloads. Prefill is one forward pass over the
> whole prompt — compute-bound, parallel across positions, and it ends at the
> first token, so it *is* TTFT. Decode is then one pass per output token,
> strictly sequential, and memory-bandwidth-bound because each step reads all the
> model weights to produce a single token. They need different tuning: prefill
> responds to chunked prefill and prefix caching, decode responds to batching and
> quantization, and neither responds to adding API replicas.
>
> The KV cache is what makes decode viable. Q belongs to the current token, but K
> and V belong to every past token and are bit-identical between steps, so they
> are cached. I measured it both ways on a toy model: identical greedy output —
> which proves it's an exact optimisation, not an approximation — with per-step
> cost flat at 16.7 ms cached versus growing and 6.5× more total work uncached.
> Without the cache it's O(n²) over the sequence; with it, O(n).
>
> The cost moves from compute to memory, and that's the trade that matters: the
> cache grows by one entry per token per sequence, so VRAM — not FLOPs — sets the
> concurrency ceiling."
