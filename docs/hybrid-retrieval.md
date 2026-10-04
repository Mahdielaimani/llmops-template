# Hybrid Retrieval and Reranking — and Why the Expected Result Did Not Happen

**Status:** LOCAL / measured. Phase 9. Produced by
`benchmarks/retrieval_modes.py` against two question sets with planted ground
truth.

> **CORRECTED IN PHASE 11.** The headline below was measured on 12 questions.
> Re-run at n=113 ([evaluation.md](evaluation.md) §5), **RRF is no longer worse
> than dense** — 0.894 against 0.876. The *mechanism* identified here held (RRF
> still damages paraphrase), and so did the other three conclusions, but the
> aggregate headline was an artifact of the sample size this document itself
> warned about in §7. Read §7 before quoting any number here.

**Headline: RRF hybrid scored *worse* than dense retrieval alone, and the
cross-encoder reranker added 98 ms for no aggregate gain.** Weighted fusion won.
The per-class breakdown shows why, and the conclusion is a changed default — not
a massaged number.

---

## 1. Why the two retrievers exist

| | Dense (bi-encoder) | Lexical (BM25) |
|---|---|---|
| Matches on | meaning | exact terms |
| Strong at | paraphrase, synonyms, intent | identifiers, codes, rare tokens |
| Blind to | an unseen exact token | anything worded differently |
| Why | `AUD-2026-014` embeds near every other code | scores 0 with no term overlap |

That is the theory. Phase 9 tests it.

### The tokenizer detail that makes BM25 work at all

```python
_TOKEN = re.compile(r"[A-Za-z0-9]+(?:[-_./][A-Za-z0-9]+)*")
```

A naive `\w+` splits `AUD-2026-014` into `aud`, `2026`, `014` — three common
tokens, and the exact-match signal is destroyed. The identifier is kept whole
*and* its parts are indexed, so a query mentioning only a fragment still hits.

**The ACL applies to the lexical index too.** A BM25 index without the same
predicate is a bypass of the Qdrant filter — same documents, different door. It
reads the payloads back out of Qdrant (`load_payloads`) rather than re-parsing
the source, so the two indexes cannot drift apart after a partial re-ingest, and
`AclPredicate.allows()` is the identical callable (ADR-019).

---

## 2. The hard set exists because Phase 8 had no headroom

Phase 8 measured **Recall@3 = 1.000 and MRR 0.958** with dense retrieval alone.
Comparing configurations against that would have measured noise.

So 12 questions were written to expose specific failure modes, each labelled
with *why* it is hard:

| Class | n | Designed to defeat | Example |
|---|---|---|---|
| `lexical` | 3 | dense | `AUD-2026-014` — a bare document code |
| `paraphrase` | 3 | BM25 | "How long do staff have to claim money back after spending it?" — zero term overlap with the policy |
| `near-duplicate` | 2 | both | two budget memos identical but for department |
| `version` | 2 | both | original vs restated Q2 revenue, same `doc_id` |
| `distractor` | 2 | both | six sibling quarterly reports sharing all vocabulary |

Labelling the class is what turns the result from a score into a mechanism.

---

## 3. Measured — the hard set

| Mode | R@1 | R@3 | R@5 | **MRR** | p50 ms | p95 ms |
|---|---|---|---|---|---|---|
| dense | 0.667 | 0.833 | 1.000 | **0.778** | 48.8 | 184.8 |
| lexical | 0.500 | 0.750 | 0.750 | 0.597 | **0.1** | 1.4 |
| hybrid (RRF) | 0.500 | 0.833 | 0.917 | **0.669** ↓ | 35.9 | 55.0 |
| **hybrid_weighted** | **0.750** | **1.000** | **1.000** | **0.861** ↑ | 22.4 | 136.0 |
| hybrid_rerank | 0.667 | 0.917 | 0.917 | **0.778** = | 124.4 | 241.0 |

Two results that contradict the standard narrative:

- **RRF hybrid is worse than dense alone** (0.669 vs 0.778).
- **The cross-encoder reranker is exactly equal to dense alone** (0.778), for
  98 ms of extra latency.

Only **weighted fusion** beat dense.

---

## 4. Why — the per-class breakdown

MRR by hardness class. This table is the actual deliverable of the phase.

| Class | dense | lexical | hybrid | weighted | rerank |
|---|---|---|---|---|---|
| `lexical` | 0.750 | **1.000** | **1.000** | **1.000** | **1.000** |
| `paraphrase` | **0.778** | **0.000** | 0.178 ↓↓ | 0.611 | 0.444 |
| `near-duplicate` | 1.000 | 1.000 | 1.000 | 1.000 | 1.000 |
| `version` | **0.375** | 0.667 | 0.500 | **0.750** | **0.750** |
| `distractor` | **1.000** | 0.417 | 0.750 | **1.000** | 0.750 |

### The predictions that held

- **BM25 beats dense on identifiers**: 1.000 vs 0.750 on `lexical`.
- **BM25 scores literally zero on paraphrase.** Not low — *zero*. With no shared
  terms there is nothing to score, which is the clearest possible demonstration
  of why a lexical index alone is not a retrieval system.
- **Dense is weakest on `version`** (0.375): v1 and v2 of the same document are
  near-identical in embedding space, so semantic similarity cannot separate
  "original" from "restated". Lexical does better because the restatement
  contains the literal word.

### Why RRF lost

`paraphrase` collapses from **0.778 (dense) to 0.178 (RRF)**. The mechanism:

```text
score(chunk) = Σ over lists of 1 / (k + rank)
```

RRF uses **only ranks**, never scores. So a BM25 hit at rank 1 contributes
`1/61` — exactly as much as a dense hit at rank 1. On a paraphrase question
BM25's top hit is *noise*, because BM25 had nothing real to match. RRF cannot
tell a confident rank 1 from a desperate one, so it promotes that noise into the
fused top-5 and displaces the correct dense hit.

Weighted fusion survives it for two reasons: `dense_weight=0.6` downweights
BM25, and min-max normalisation collapses BM25's contribution toward zero when
its score range is tiny — which is precisely the signal RRF discards.

**RRF is not wrong in general.** It is score-calibration-free, which is its whole
appeal, and that is an advantage when both retrievers return many comparable
candidates. On an 18-chunk corpus where one retriever frequently returns 1–3
weak hits, discarding the scores discards the information that those hits are
weak. See ADR-003.

### Why the reranker did not help

`ms-marco-MiniLM-L-6-v2` reranks the **fused** list, so it inherits RRF's damage
— it can only reorder what it is handed. It recovered `lexical` (1.000) and
`version` (0.750), and improved paraphrase over RRF (0.178 → 0.444), but could
not get back to dense's 0.778 because the right chunk was no longer ranked where
it could win.

It is also trained on MS MARCO web passages, not financial documents dense with
codes and figures. A domain-matched reranker, or reranking the *dense* list
rather than the fused one, are both untested alternatives.

---

## 5. Latency — and a prediction I got wrong

Mean per stage, milliseconds:

| Mode | embed | dense | lexical | fusion | rerank | acl re-check |
|---|---|---|---|---|---|---|
| dense | 7.08 | 63.30 | — | 0.00 | — | 0.12 |
| lexical | — | — | **0.24** | 0.00 | — | — |
| hybrid | 6.48 | 29.56 | 0.21 | 0.13 | — | 0.11 |
| hybrid_weighted | 10.23 | 23.72 | 0.24 | 0.11 | — | 0.13 |
| hybrid_rerank | 10.70 | 25.40 | 0.32 | 0.14 | **98.02** | 0.11 |

- **BM25 is essentially free**: 0.24 ms against 25 ms for the vector search.
  In-process on 18 chunks, so this will not hold at scale — but it does mean
  that *adding* lexical retrieval costs nothing, and the only question is
  whether the fusion helps.
- **The ACL re-check is 0.11–0.13 ms**, about 0.3% of retrieval. Two independent
  enforcement points for free.
- **I predicted the reranker at 200–400 ms. It measured 98 ms.** Recorded in
  `docs/rag.md` §5 before measuring; the prediction was ~3× too pessimistic.
  Still the dominant stage — 79% of `hybrid_rerank` latency — but less dominant
  than claimed.

Against **SLO-5 (retrieval p95 ≤ 600 ms)**: every mode passes. `hybrid_rerank`
at 241 ms p95 uses 40% of the budget, which is affordable but is no longer free.

---

## 6. The easy set — nothing regressed

| Mode | R@1 | R@3 | MRR |
|---|---|---|---|
| dense | 0.917 | 1.000 | 0.958 |
| lexical | 0.833 | 1.000 | 0.917 |
| hybrid | **1.000** | 1.000 | **1.000** |
| hybrid_weighted | **1.000** | 1.000 | **1.000** |
| hybrid_rerank | **1.000** | 1.000 | **1.000** |

Every hybrid variant reaches a perfect score. This both confirms no regression
and confirms why the hard set was necessary: **on the easy set, hybrid looks
like a clear win and the reranker looks free.** Measuring only this would have
produced exactly the wrong conclusion.

---

## 7. The honest caveat: n = 12

**One question is worth 0.083 MRR.** So:

- weighted (0.861) beating dense (0.778) is **one question**
- RRF (0.669) losing to dense is **one or two questions**
- rerank equalling dense is a genuine tie

Differences below roughly 0.15 MRR are not distinguishable from noise at this
sample size. What *is* trustworthy is the per-class pattern, because it matches
a mechanism that is independently explainable — BM25 scoring zero on paraphrase
is not a sampling artifact.

**Phase 11 needs a materially larger evaluation set** (target: 100+ questions,
as `business-requirements.md` §7 assumes) before any of these numbers should be
quoted as a configuration decision. This document is a mechanism study, not a
benchmark.

---

## 8. ACL holds in every mode

| Mode | contractor | analyst-emea | cfo-office |
|---|---|---|---|
| dense | clean | clean | yes (authorized) |
| lexical | clean | clean | yes |
| hybrid | clean | clean | yes |
| hybrid_weighted | clean | clean | yes |
| hybrid_rerank | clean | clean | yes |

**Violations: 0.** Adding a second retriever, two fusion strategies and a
reranker did not open a path to the confidential memo — because the predicate is
applied inside *both* retrievers' candidate selection, not after fusion. Fusing
first and filtering second would have been the bug.

---

## 9. Decisions taken from this

1. **Default mode is `hybrid_weighted`**, not `hybrid_rerank`. It scored best on
   the hard set, best-equal on the easy set, and costs 100 ms less. Changed on
   evidence, pending Phase 11's larger set.
2. **RRF is kept as a mode, not deleted.** It is the right choice at larger
   candidate pools and the comparison must stay runnable (ADR-003).
3. **Reranking is kept but off by default** (ADR-004). It clearly helps the
   classes it should; the open question is whether reranking the dense list
   rather than the fused one recovers paraphrase.
4. **Mode is a request parameter** so Phase 11 evaluates all five against
   identical questions and ACL. In production it is configuration, not
   client-chosen.

---

## 10. Reproduce it

```bash
docker compose --profile core up -d qdrant
uv run python -m rag_service.corpus      # 12 docs + 12 facts + 12 hard questions
uv run python -m rag_service.ingest
uv run python benchmarks/retrieval_modes.py   # exits non-zero on any ACL violation

curl -s localhost:8070/rag -H 'content-type: application/json' -H 'x-user: cfo-office' \
  -d '{"question":"AUD-2026-014","mode":"dense","top_k":3}' | jq '.citations[].doc_id'
# then the same with "mode":"lexical" — the difference is the whole point
```

---

## 11. Interview answer

> "Dense and lexical retrieval fail in opposite directions, so the textbook
> answer is to fuse them and rerank. I measured it on a question set built to
> expose each failure mode, and the textbook answer lost.
>
> The mechanism held exactly as predicted per class: BM25 beat dense on bare
> document codes, and scored *zero* on paraphrase — not low, zero, because with
> no shared terms there is nothing to match. Dense was weakest on version
> disambiguation, since v1 and v2 of the same document are nearly identical in
> embedding space.
>
> But Reciprocal Rank Fusion scored worse than dense alone, and the cross-encoder
> came out exactly equal for 98 ms of extra latency. The reason is that RRF uses
> only ranks: a BM25 hit at rank 1 contributes the same as a dense hit at rank 1,
> so on a paraphrase question it promotes BM25's noise — paraphrase MRR fell from
> 0.78 to 0.18. Weighted fusion survived because normalising the scores collapses
> BM25's contribution when its score range is tiny, which is precisely the
> information RRF throws away. So RRF's score-calibration-free property, which is
> its main selling point, is also why it fails on a small corpus where one
> retriever often returns a few weak hits.
>
> I changed the default to weighted fusion on that evidence and kept RRF as a
> mode, because it is the right choice at larger candidate pools. And I'd flag
> the sample size: twelve questions means one question is 0.083 MRR, so the
> aggregate differences are one or two questions. The per-class pattern is what I
> trust, because it matches a mechanism I can explain. The aggregate needs a
> hundred questions before it decides anything."

---

## 12. Related

- [rag.md](rag.md) — the ingestion pipeline and the Phase 8 baseline this builds on
- [ADR-003](adr/ADR-003-hybrid-retrieval.md) — fusion strategy, and when RRF is right
- [ADR-004](adr/ADR-004-reranking.md) — reranker choice and why it is off by default
- [ADR-019](adr/ADR-019-authorization-model.md) — why the ACL goes inside both retrievers
- [business-requirements.md](business-requirements.md) §4 — SLO-5, Q-1, Q-2
