# ADR-003: Hybrid retrieval, with weighted fusion as the default rather than RRF

**Status:** Accepted
**Date:** 2026-10-03
**Phase:** 9

## Context
Dense and lexical retrieval fail in opposite directions. Measured on a question
set built to expose each (`docs/hybrid-retrieval.md` §4):

| Hardness class | dense MRR | BM25 MRR |
| --- | --- | --- |
| identifiers (`AUD-2026-014`) | 0.750 | **1.000** |
| paraphrase, no shared terms | **0.778** | **0.000** |
| version disambiguation | 0.375 | 0.667 |

BM25 scoring *zero* on paraphrase, and dense losing on bare document codes, is
the case for combining them. BM25 also costs 0.24 ms against 25 ms for the
vector search, so adding it is effectively free — the only real question is how
to merge the two ranked lists.

The expected answer was Reciprocal Rank Fusion, which needs no score
calibration. **Measured, RRF scored worse than dense retrieval alone**
(MRR 0.669 against 0.778).

## Decision
Run both retrievers always. Offer five modes and default to
**weighted fusion with `dense_weight = 0.6`**.

```text
dense            vector only
lexical          BM25 only
hybrid           RRF(dense, lexical)              kept for comparison
hybrid_weighted  min-max normalised weighted sum  DEFAULT
hybrid_rerank    fusion then cross-encoder        see ADR-004
```

The ACL predicate is applied inside **both** retrievers' candidate selection,
never after fusion. Fusing first and filtering second would reintroduce every
post-filter failure in `docs/security.md` §2.

## Why RRF lost, specifically
`score(chunk) = sum of 1 / (k + rank)` uses **only ranks**. A BM25 hit at rank 1
contributes `1/61`, identical to a dense hit at rank 1. On a paraphrase query
BM25 had nothing real to match, so its rank-1 hit is noise — and RRF cannot
distinguish a confident rank 1 from a desperate one. Paraphrase MRR fell from
0.778 to 0.178.

Weighted fusion survives because min-max normalisation collapses BM25's
contribution toward zero when its score range is tiny, which is exactly the
information RRF discards by design.

**This is not a general claim about RRF.** Being calibration-free is its
strength when both retrievers return many comparable candidates. On an 18-chunk
corpus where one retriever routinely returns one to three weak hits, discarding
scores discards the signal that those hits are weak. Revisit at a realistic
corpus size — which is why RRF stays implemented rather than deleted.

## Alternatives considered
| Option | Why not (as default) |
| --- | --- |
| **RRF** | measured worse than dense alone at this corpus size, for the reason above; retained as a mode and expected to win back at scale |
| Dense only | loses on identifiers (0.750 vs 1.000) and version disambiguation; BM25 is free to add |
| BM25 only | scores 0.000 on paraphrase — not a retrieval system on its own |
| Qdrant native sparse vectors / BM42 | removes the separate index and scales properly; rejected for now because an in-process BM25 keeps the lexical stage visible in our own latency breakdown, which is the point of the lab. The right move at scale |
| Elasticsearch / OpenSearch | the production answer for lexical at scale; another stateful container on a disk at 97% |
| Learned sparse (SPLADE) | strong, and bridges both regimes; needs a torch-based model, which ADR-020 rules out here |
| Tuning `dense_weight` per query | plausible, but with n=12 any tuning would fit noise |

## Trade-offs
+ covers both failure modes, and BM25 costs 0.24 ms
+ weighted fusion measured best on the hard set and best-equal on the easy set
+ mode is a request parameter, so Phase 11 evaluates all five identically
− `dense_weight = 0.6` is a tuned constant, tuned on 12 questions. It is a
  versioned retrieval-config artifact (ADR-015 class 5), not a constant
− min-max normalisation has its own failure mode: a retriever returning a single
  result maps it to 1.0 regardless of quality. RRF has no such flaw, and that
  asymmetry is why both stay available
− the in-process BM25 index is rebuilt at startup from Qdrant payloads, so
  startup time scales with corpus size

## Consequences
- `RetrievalPipeline` owns all five modes, so a comparison measures strategies
  rather than five implementations.
- `LexicalIndex` reads payloads back out of Qdrant rather than re-parsing the
  source documents, so the two indexes cannot drift after a partial re-ingest.
- The BM25 tokenizer keeps identifiers whole *and* indexes their parts; a naive
  `\w+` would split `AUD-2026-014` into three common tokens and destroy the
  signal this index exists to provide.
- `dense_weight` and the fusion mode join the retrieval-config hash, so changing
  either is a tracked change.
- **Phase 11 must re-run this on 100+ questions before these numbers decide
  anything.** One question is 0.083 MRR here.
