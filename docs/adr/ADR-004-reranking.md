# ADR-004: Cross-encoder reranking, available but off by default

**Status:** Accepted
**Date:** 2026-10-03
**Phase:** 9

## Context
A bi-encoder embeds the query and the passage separately, so it can only measure
vector proximity. A cross-encoder reads both *together* and can represent "this
passage answers this question" — more accurate, and far too slow to run over a
corpus, so it reranks a small candidate set.

That is the theory. Measured on the hard question set
(`docs/hybrid-retrieval.md` §3):

| Mode | MRR | p50 latency |
| --- | --- | --- |
| dense | 0.778 | 48.8 ms |
| hybrid (RRF) | 0.669 | 35.9 ms |
| **hybrid_rerank** | **0.778** | **124.4 ms** |
| hybrid_weighted | **0.861** | 22.4 ms |

**The reranker came out exactly equal to dense retrieval alone, for 98 ms of
additional latency.**

## Decision
Use `Xenova/ms-marco-MiniLM-L-6-v2` (ONNX, 80 MB) via fastembed, over the fused
candidates, top-20 in and top-5 out. Keep it as a mode. **Do not make it the
default** until it demonstrates a gain on a larger evaluation set.

## Why it did not help here
1. **It inherits the fusion's damage.** It reranks the *fused* list, so RRF's
   promotion of BM25 noise on paraphrase queries is already baked in — a
   reranker can only reorder what it is handed. Paraphrase recovered from 0.178
   to 0.444, but could not reach dense's 0.778.
2. **Domain mismatch.** MS MARCO is web-search passages. This corpus is
   financial documents dense with codes, figures and near-duplicate siblings.
3. **It did help where it should.** `lexical` 1.000 and `version` 0.750 — the
   classes where reading query and passage together is the only way to separate
   "original" from "restated".

Untested and cheap to try: rerank the **dense** list instead of the fused one,
which would avoid inheriting the fusion's errors entirely.

## Latency, and a prediction corrected
`docs/rag.md` §5 predicted 200–400 ms for a CPU cross-encoder over 20
candidates, before measuring. **Measured: 98 ms mean.** The prediction was about
3× too pessimistic. It remains the dominant stage — 79% of `hybrid_rerank`
latency — and consumes 40% of the SLO-5 budget (241 ms p95 against 600 ms), so
it is affordable but no longer free.

## Alternatives considered
| Option | Why not (now) |
| --- | --- |
| `ms-marco-MiniLM-L-12-v2` (120 MB) | better quality, roughly double the latency; worth evaluating in Phase 11 |
| `bge-reranker-base` (1.04 GB) | likely the best quality available here and probably domain-appropriate; rejected on disk at 97%, reserved for vLLM (ADR-020). The first thing to try once disk is freed |
| `jina-reranker-v1-turbo-en` (150 MB) | plausible alternative, untested |
| A torch cross-encoder via sentence-transformers | reopens ADR-020's ~1 GB problem |
| LLM-as-reranker | strongest and slowest; it is also a judge, so it belongs with Phase 12 where its bias can be documented |
| No reranker at all | then version disambiguation stays at 0.375–0.750, and that is the class this corpus most needs right, since it contains a restatement |
| Reranking on GPU | the production answer; blocked with Phases 6 and 7 |

## Trade-offs
+ clearly recovers the classes a bi-encoder cannot separate
+ ONNX keeps ADR-020 intact — no torch
+ 98 ms is affordable within SLO-5
− no aggregate gain measured, so enabling it by default would cost latency for
  nothing on current evidence
− n=12, so "equal to dense" is a genuine tie rather than a measured absence of
  effect. This conclusion is provisional on Phase 11
− reranking the fused list couples this decision to ADR-003's fusion choice

## Consequences
- Default mode is `hybrid_weighted`; `hybrid_rerank` is opt-in per request.
- The reranker model name joins the retrieval-config hash (ADR-015 class 5).
- Phase 11 re-evaluates on 100+ questions, and additionally tests reranking the
  dense list rather than the fused one.
- Once disk is freed, `bge-reranker-base` is the first upgrade to evaluate.
- The model is pre-downloaded into the image, like the embedder, so startup does
  not pay a download and the readiness probe does not flap.
