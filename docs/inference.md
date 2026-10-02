# Transformer Inference — Mechanics

**Status:** LOCAL / educational. Phase 4. Every number is produced by a script in
`inference/experiments/`; nothing here is asserted from memory.

**THIS IS A SIMPLIFICATION.** Attention and the forward pass are implemented in
numpy with random weights, to make the shapes and arithmetic visible. There is no
KV cache (Phase 5), no fused kernel, no batching scheduler, no GPU (Phase 6, 8).
Generated text is therefore noise — the mechanics are the deliverable, not the
output. The **tokenizer is real** GPT-2 BPE, because tokenization is where this
project's cost model begins.

Geometry used throughout is GPT-2 small's, so the figures are not invented:
`vocab 50257 · d_model 768 · 12 layers · 12 heads · d_head 64 · d_ff 3072 · max_pos 1024`.

```text
text ── tokenizer ──> ids ── embedding lookup ──> hidden states
                                                       │
                                        ┌──────────────┴──────────────┐
                                        │  x12 blocks:                │
                                        │    LayerNorm                │
                                        │    Q,K,V = x@Wq, x@Wk, x@Wv │
                                        │    scores = QKᵀ/√d_head     │
                                        │    + causal mask, softmax   │
                                        │    out = (weights@V)@Wo     │
                                        │    + residual               │
                                        │    LayerNorm, MLP, residual │
                                        └──────────────┬──────────────┘
                                                       │
                           final LayerNorm ── @ wteᵀ ──> logits (n, 50257)
                                                       │
                              take LAST row ── softmax ──> sample ──> one token
```

Run them:
```bash
uv run python inference/experiments/01_tokenizer.py
uv run python inference/experiments/02_embeddings.py
uv run python inference/experiments/03_attention.py
uv run python inference/experiments/04_sampling.py
uv run python inference/experiments/05_forward_pass.py
```

---

## 1. Tokenization — and the finding that changes the capacity model

A token is neither a word nor a character. Measured with the real GPT-2 tokenizer:

| Text | chars | tokens | **chars/token** |
| --- | --- | --- | --- |
| `The company reported strong growth in the second quarter.` | 57 | 10 | **5.70** |
| `Q2 FY2026 revenue was EUR 1,234,567.89, up 12.4% against plan of EUR 1,098,000.00.` | 82 | 35 | **2.34** |
| `Invoice INV-2026-0004471 for cost centre CC-88213-EMEA, PO 4500931882.` | 70 | 28 | **2.50** |

**Financial text costs 2.4× more tokens per character than prose.** One amount:

```text
1,234,567.89  ->  ['27526','352','11','24409','11','20','3134','13','4531']   9 tokens
```

BPE was trained on web text, so rare digit groups fragment. An invoice identifier
is worse: `['Inv','oice','ĠINV','-','20','26','-','000','44','71']`.

### Consequence for this platform

`docs/metrics-and-capacity.md` §6 models `P` as system + template + k chunks +
query. A "400-token chunk" of financial text holds roughly **940 characters**,
where 400 tokens of prose holds **2 280**. At a fixed context budget you retrieve
**2.4× less information** from financial documents than the generic estimate
implies. Measured budget for a realistic RAG request:

```text
system prompt         12 tokens
user query            11 tokens
5 financial chunks   700 tokens  (140 each)
                     ---
prefill P            723 tokens
```

`P` feeds prefill FLOPs (`2·N·P` plus a `P²` attention term) and API cost
directly. This is why the chunking parameters are a versioned artifact
(`docs/versioning.md`, class 5) rather than a tuning detail.

### Whitespace and casing are part of the token

```text
'revenue'   -> [260, 4080]          ['re','venue']
' revenue'  -> [6426]               ['Ġrevenue']
'Revenue'   -> [3041, 4080]         ['Re','venue']
' Revenue'  -> [20197]              ['ĠRevenue']
'REVENUE'   -> [2200, 28290, 8924]  ['RE','VEN','UE']
```

The leading space is encoded *inside* the token (`Ġ`), so ` revenue` is one token
and `revenue` is two. Five renderings of one word, five different id sequences.

Two operational consequences: a prompt template's whitespace is load-bearing
(hence prompt versioning, `docs/versioning.md` class 2), and concatenating
retrieved chunks without care changes the tokenization at every join.

Round-trip is lossless: `decode(encode(text)) == text`.

---

## 2. Embeddings — discrete becomes continuous

| Table | Shape | Parameters |
| --- | --- | --- |
| `wte` token embeddings | (50257, 768) | 38 597 376 |
| `wpe` position embeddings | (1024, 768) | 786 432 |

`wte` alone is **77 MB in fp16** — about 31% of GPT-2 small. Embedding is a row
lookup, `wte[ids]`; no matmul, no FLOPs worth counting.

```text
ids       (25,)
tok_emb   (25, 768)      wte[ids]
pos_emb   (25, 768)      wpe[0..24]
x         (25, 768)      tok_emb + pos_emb
```

### Position is added, and that is not cosmetic

Attention is permutation-invariant. Measured, with the token order reversed:

| | same result for reversed input? |
| --- | --- |
| without position embeddings | **True** |
| with position embeddings | **False** |

Without `wpe`, `revenue was 100` and `100 was revenue` are the same input.

### LayerNorm fixes scale, it does not shrink

```text
norm before LayerNorm:    0.612
norm after  LayerNorm:   27.432      (~sqrt(768) = 27.7)
output mean: -1.42e-18   std: 0.9901
```

Note it went **up**. LayerNorm sets each token vector to zero mean and unit
variance per dimension, so its norm lands near `√d_model` regardless of input
scale. That scale-invariance is the point: the residual stream accumulates across
12 layers, and without renormalisation before each sub-block the attention
softmax saturates (§3).

One hidden state is 768 floats = 1 536 B in fp16. Activations are transient and
freed after the pass; K and V are **not**, during generation — that is the cache.

---

## 3. Attention — Q, K, V, and why only two are cached

Three projections of the same input, each (768, 768):

```text
x (15,768) @ Wq -> Q (15,768)
x (15,768) @ Wk -> K (15,768)
x (15,768) @ Wv -> V (15,768)
```

| | Meaning | Belongs to |
| --- | --- | --- |
| **Q** | what am I looking for | the **current** token |
| **K** | what can I be found by | **every past** token |
| **V** | what I contribute | **every past** token |

**That asymmetry is the entire justification for the KV cache.** At each decode
step Q is computed for the one new token, while K and V for all previous tokens
are bit-identical to the previous step. Q has nothing to reuse; K and V have
everything.

### Heads are a reshape, not extra parameters

`768 dims → 12 heads × 64 dims`, giving Q per head `(12, 15, 64)`. Each head
attends independently over the same tokens.

### The `√d_head` divisor is load-bearing

| | std | max abs | softmax peak |
| --- | --- | --- | --- |
| raw `QKᵀ` | 8.015 | 32.029 | **0.8912** |
| `/√64` | 1.002 | 4.004 | **0.2003** |

Unscaled, dot products grow with `d_head`, softmax saturates toward one-hot, and
the layer degenerates into copying a single token. The divisor keeps the
distribution soft enough to mix information.

### The n² term, concretely

15 tokens → **225** scores per head → **2 700** across 12 heads. Double the
context and this quadruples. That is the `P²` term in the prefill formula.

### The causal mask

```text
mask[0,  :5] = [  0. -inf -inf -inf -inf]   row 0 sees only itself
mask[-1, :5] = [  0.    0.    0.    0.   0.]   last row sees all history
```

Verified: every row of the softmax sums to **1.0**, and the upper triangle is
**exactly 0**.

The mask does two jobs. In training it stops the model seeing the future. At
inference it is what makes caching *valid*: because a past token can never attend
forward, its attention output does not change when a new token arrives.

### Output is a weighted sum of V

```text
weights (12,15,15) @ V (12,15,64) -> (12,15,64)
merge heads -> (15,768)  @ Wo -> (15,768)
```

Same shape in, same shape out. That invariance is what lets 12 blocks stack.

### What the cache buys — measured formula

```text
KV per token per sequence = 2 (K,V) · 12 layers · 12 heads · 64 dim · 2 B
                          = 36 864 B = 36 KiB
```

| Context | KV per sequence |
| --- | --- |
| 512 | 18.9 MB |
| 2 048 | 75.5 MB |
| 4 096 | 151.0 MB |

Without a cache, generating token `t` recomputes K and V for all `t−1` previous
tokens: `O(t)` per step, `O(t²)` for the sequence. With it, `O(1)` projections
per step — the cost moves from compute to **memory**, which is exactly why decode
is bandwidth-bound and why VRAM sets the concurrency ceiling.

GPT-2's 36 KiB/token is **not** the number for the lab's target model.
Qwen2.5-7B uses grouped-query attention — 4 KV heads instead of 28 — giving
56 KiB/token at 28 layers (`docs/metrics-and-capacity.md` §5). GQA exists
precisely to cut this term.

---

## 4. Logits and sampling

The unembedding is the widest matmul in the model:

```text
hidden (768,) @ W_unembed (768, 50257) -> logits (50257,)
```

50 257 scores for **one** position. Logits are unbounded reals, not
probabilities, and only their differences matter — verified:
`softmax(logits) == softmax(logits + 1000)` → **True**.

> Random Xavier weights give a logit std of **0.183**, which is nearly uniform
> over 50 k tokens and makes every temperature look identical. A trained GPT-2
> produces a final logit std nearer **4**, so the script rescales to that and says
> so. Labelled rather than hidden, because the un-rescaled table taught the
> opposite of the truth.

### Temperature is distribution sharpness, not a creativity dial

| T | max p | top-10 mass | entropy | effective choices |
| --- | --- | --- | --- | --- |
| 0.01 | 0.99702 | 1.00000 | 0.020 | **1** |
| 0.20 | 0.50561 | 0.99995 | 1.097 | 3 |
| 0.70 | 0.20556 | 0.81028 | 2.980 | 20 |
| 1.00 | 0.10484 | 0.50751 | 4.712 | 111 |
| 1.50 | 0.02912 | 0.17203 | 7.311 | 1 496 |
| 3.00 | 0.00188 | 0.01415 | 9.918 | 20 291 |

`T → 0` approaches greedy. `T = 1` is the raw distribution. `T = 3` makes 20 000
tokens plausible. For grounded financial answers the lab defaults to `0.2`
(`config.py`).

### Determinism

| | result |
| --- | --- |
| greedy, 5 runs | `[14247] × 5` — **identical** |
| sampled, 5 different seeds | `[32014, 25740, 13166, 4302, 47408]` — **different** |
| sampled, seed 42, 3 runs | `[38909] × 3` — **identical** |

Reproducibility needs the seed **and** identical logits. Logits shift with batch
composition, kernel version and hardware, so even `temperature=0` on a real
server is not bit-reproducible across runs.

This is the root of a platform decision: quality must be measured statistically
over a dataset (Phase 11) and regression-tested against a baseline rather than a
fixture (Phase 14). Contrast
`docs/classical-ml-vs-llm-serving.md` §4, where six identical calls return a
byte-identical probability.

### top-k is fixed, top-p adapts

| Distribution | top-k=50 | top-p=0.9 |
| --- | --- | --- |
| ordinary | 50 tokens | **247** tokens |
| deliberately confident | 50 tokens | **1** token |

top-k keeps 50 either way — including 49 the model had all but ruled out. top-p
collapses to the few that carry the mass. For grounded answers the adaptive cut
is the safer default.

---

## 5. The full pass, and where the FLOPs go

### Parameter budget from the geometry

```text
per block = 4·768·768 (QKVO) + 2·768·3072 (MLP) = 7 077 888
embeddings                                        39 383 808
12 blocks                                         84 934 656
total                                            124 318 464   (~124M)
   fp16  249 MB      int4  62 MB
```

Matches GPT-2 small's published ~124M.

### Shapes end to end

```text
text              'Q2 FY2026 revenue was EUR 1,234,567.89, up 12.4% against plan.'
ids               (25,)
+ embeddings      (25, 768)
after block 0     (25, 768)     <- invariant
after block 1     (25, 768)
final LayerNorm   (25, 768)
@ wteᵀ (tied)     (25, 50257)
take last row     (50257,)
softmax           sums to 1.000000
```

### Prefill computes every position; decode needs one

Prefill produced logits for all 25 positions. **24 of those 25 rows are thrown
away.** Training needs all of them (next-token loss at every position);
inference needs the last. That difference is why prefill is one large batched
matmul and decode is a thin one, and why the two phases have unrelated
performance profiles — measured in Phase 5.

### FLOPs: linear term vs quadratic term

| P | dense `2·N·P` | attention `2·L·P²·d` | attention share |
| --- | --- | --- | --- |
| 25 | 6.22 GFLOP | 0.01 GFLOP | **0.2%** |
| 512 | 127.30 GFLOP | 4.83 GFLOP | **3.7%** |
| 3 000 | 745.91 GFLOP | 165.89 GFLOP | **18.2%** |

The attention term is quadratic while the dense term is linear, so its *share*
grows with context. On a 7B model the dense term dominates far longer, which is
why a 3 000-token context is still prefill-dominated there.

### Measured wall time — and a correction

| P | best of 3 | ms/token | normalised |
| --- | --- | --- | --- |
| 128 | 110.6 ms | 0.864 | 1.00× |
| 256 | 638.9 ms | 2.496 | 2.89× |
| 512 | 396.2 ms | 0.774 | 0.90× |
| 1 024 | 889.9 ms | 0.869 | 1.01× |

**Roughly linear**, and that is the correct result. The `P=256` row is
measurement noise on a laptop running five containers, not a scaling effect —
three of four points sit within 12% of each other.

An earlier version of this script took three unwarmed samples and concluded
"superlinear, as the P² term predicts". The data did not support it, and §4
explains why it could not: below `P=512` the attention term is under 4% of total
FLOPs, so the quadratic component is invisible. Recorded here because publishing
the first table would have put a false claim into the lab's own reference
material — the same failure mode as the 43 ms port-proxy trap in Phase 3.

---

## 6. What carries into the rest of the platform

| Concept here | Where it becomes operational |
| --- | --- |
| `P` from real tokenization | prefill FLOPs, TTFT, API cost — `metrics-and-capacity.md` §2, §6, §7 |
| financial text at 2.34 chars/token | chunk sizing, context budget, Phase 9 |
| K/V reusable, Q not | KV cache — Phase 5 |
| 36 KiB/token (GPT-2) vs 56 KiB (Qwen GQA) | concurrency ceiling — §5 of the capacity doc |
| causal mask makes caching valid | why prefix caching is correct at all |
| prefill all positions / decode one | the two-phase performance split — Phase 5, 7 |
| non-determinism | statistical evaluation, Phases 11–14 |
| temperature and top-p | guardrails and grounded-answer defaults, Phase 28 |
| whitespace changes ids | prompt versioning — `versioning.md` class 2 |

## 7. Interview answer

> "Text becomes ids through a BPE tokenizer — and that step already matters
> operationally: I measured financial text at 2.34 characters per token against
> 5.70 for prose, so a fixed context budget retrieves 2.4× less information from
> financial documents, which changes chunk sizing and cost per request. Ids index
> an embedding table, positions are added because attention is otherwise
> permutation-invariant, and then each block projects the hidden state into Q, K
> and V. Q belongs to the current token; K and V belong to every past token and
> are bit-identical between decode steps — that asymmetry is the whole reason a
> KV cache exists and why it, not compute, sets the concurrency ceiling. Scores
> are divided by √d_head or the softmax saturates to one-hot; the causal mask
> prevents attending forward, which is also what makes caching valid. The final
> hidden state is projected to one logit per vocabulary entry, and only the last
> position's row is used at inference — prefill computes all of them and discards
> the rest, which is why prefill and decode have completely different performance
> characteristics."
