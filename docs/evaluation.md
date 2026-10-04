# Evaluation

**Status:** LOCAL / PRODUCTION-LIKE. Phase 11. All numbers from
`uv run llmops-eval`, scored against planted ground truth — no judge involved.

**Headline: Phase 9's main conclusion was wrong, and the sample size was why.**
With n=113 instead of n=12, RRF fusion no longer loses to dense retrieval. The
other three Phase 9 conclusions held. Details in §5.

---

## 1. Why a 113-item dataset came first

Phase 9 compared five retrieval configurations on 12 questions and recorded the
limitation in `docs/hybrid-retrieval.md` §7:

> One question is worth 0.083 MRR. Differences below roughly 0.15 MRR are not
> distinguishable from noise at this sample size. **Phase 11 needs a materially
> larger evaluation set before any of these numbers should be quoted as a
> configuration decision.**

So the first deliverable of Phase 11 is the dataset, not the harness.

| | Phase 9 | Phase 11 |
|---|---|---|
| Items | 12 | **113** |
| One question is worth | 0.0833 MRR | **0.0088 MRR** |
| Resolution | ~2 questions | ~0.01 MRR |

### Generated, not hand-written

`rag_service.evalset` builds the set from the same seeded corpus that produced
the documents. Three reasons:

1. **Ground truth cannot drift** — answers are derived from the same generator,
   so a corpus change regenerates both sides together.
2. **Reproducible** — the set is content-hashed (`36b90db3c48d`), and a metric
   without its dataset version is meaningless (ADR-015 class 8).
3. **Coverage is countable** rather than hoped for.

### Composition

| Hardness class | n | Targets |
|---|---|---|
| **paraphrase** | 42 | dense retrieval's strength, BM25's blind spot |
| lexical | 24 | exact identifiers — BM25's strength |
| distractor | 15 | six sibling quarterly reports sharing vocabulary |
| planted | 12 | exact-figure ground truth, the only items with a checkable answer |
| **version** | 9 | original vs restated — dense retrieval's weakest class |
| near-duplicate | 6 | two budget memos identical but for department |
| **unanswerable** | 5 | abstention is the correct behaviour |

Weighted deliberately toward **paraphrase** and **version**: Phase 9 showed
`lexical` is easy for BM25 and `near-duplicate` easy for everything, so adding
more of those inflates `n` without improving resolution.

All four classifications appear (public 11, internal 93, restricted 5,
confidential 4), so **the same dataset measures retrieval quality and ACL
enforcement** rather than needing two.

---

## 2. Metrics, and why each one is there

| Metric | Answers |
|---|---|
| Recall@k | did the right document appear in the top k |
| MRR | how high, averaged — harsh on anything below rank 1 |
| nDCG@5 | how high, discounted gently — at rank 3, MRR says 0.33 and nDCG says 0.50, and with a five-chunk context window rank 3 is not a third as good as rank 1 |
| **context precision** | of what we put in the prompt, how much was useful — paid for twice, in prefill tokens and in the chance the model answers from the wrong document |
| context recall | did the evidence make it into the prompt at all |
| **ACL violations** | did any identity retrieve above its clearance — **a hard gate** |
| abstention / false-abstention rate | declared, measured from Phase 15 |

Two deliberate choices:

- **Unanswerable items are excluded from retrieval metrics.** There is nothing to
  recall, and including them would quietly reward or punish abstention inside a
  retrieval number. The per-class table prints `n/a` rather than `0.000`, because
  a zero there reads as a failure when the metric is simply undefined.
- **Abstention is recorded as not-measured, not guessed.** Retrieval cannot
  abstain; that is a generation behaviour and arrives with Phase 15. The field
  exists and is honest about being empty.

---

## 3. Baseline — `hybrid_weighted`, n=113

```text
dataset  1.0.0 (36b90db3c48d)   n=113
index    4a0a78edcfa9           embed=BAAI/bge-small-en-v1.5
mode     hybrid_weighted        top_k=5  candidates=20
```

| Metric | Value |
|---|---|
| Recall@1 | **0.8796** |
| Recall@3 | **1.0000** |
| Recall@5 | 1.0000 |
| **MRR** | **0.9383** |
| nDCG@5 | 0.9544 |
| context precision | 0.2796 |
| context recall | 1.0000 |
| **ACL violations** | **0** |
| Missed | **0** |
| Latency p50 / p95 | 19.8 ms / 25.4 ms |

**Gates pass**: Q-1 (Recall@5 ≥ 0.80) → 1.000; Q-2 (MRR ≥ 0.65) → 0.938;
Q-6 (ACL = 0) → 0.

### On that context precision of 0.28

With one relevant document and `top_k=5`, the ceiling is bounded by how many
chunks of the right document exist. 0.28 means roughly 1.4 of 5 chunks were
relevant — so **about 72% of every prompt is irrelevant context**, paid for in
prefill FLOPs (`metrics-and-capacity.md` §2) and in distraction risk. The obvious
lever is a smaller `top_k`, and the obvious cost is recall. That trade is
measurable now and is Phase 14 work.

### Per class

| Class | n | R@1 | R@5 | MRR | ctx prec |
|---|---|---|---|---|---|
| planted | 12 | 1.000 | 1.000 | 1.000 | 0.233 |
| near-duplicate | 6 | 1.000 | 1.000 | 1.000 | 0.200 |
| lexical | 24 | 0.958 | 1.000 | 0.979 | 0.300 |
| paraphrase | 42 | 0.881 | 1.000 | 0.936 | 0.314 |
| **version** | 9 | **0.778** | 1.000 | 0.889 | 0.200 |
| **distractor** | 15 | **0.667** | 1.000 | 0.833 | 0.267 |
| unanswerable | 5 | n/a | n/a | n/a | n/a |

Recall@5 is 1.000 everywhere — the right document is always in the prompt. What
varies is **rank**, and the two weakest classes are the two the corpus was built
to be hard at: distinguishing a restated document from its original, and picking
one of six near-identical quarterly reports.

---

## 4. ACL — the hard gate, 678 queries

Every identity against every question: 6 × 113.

| Identity | Clearance | Violations |
|---|---|---|
| contractor | public | **0** |
| analyst-emea | internal | **0** |
| analyst-apac | internal | **0** |
| auditor | restricted | **0** |
| auditor-expired | restricted (expired) | **0** |
| cfo-office | confidential | **0** |

**Q-6: 0 violations.** A violation fails the run on its own regardless of every
other number, and there is a test asserting that a run with perfect retrieval
and one unauthorized retrieval still fails.

Measuring ACL across *all* identities matters: a single-identity run cannot show
"no identity ever retrieves above its clearance", which is the actual property.

---

## 5. Re-running Phase 9 — one conclusion reversed, three held

Same five configurations, now n=113:

| Mode | MRR @ n=12 | **MRR @ n=113** | R@1 | nDCG@5 | p50 ms |
|---|---|---|---|---|---|
| dense | 0.778 | 0.876 | 0.806 | 0.907 | 19.8 |
| lexical | 0.597 | 0.812 | 0.685 | 0.854 | **0.1** |
| **hybrid (RRF)** | **0.669** | **0.894** | 0.806 | 0.919 | 19.9 |
| **hybrid_weighted** | **0.861** | **0.938** | **0.880** | **0.954** | 19.4 |
| hybrid_rerank | 0.778 | 0.867 | 0.759 | 0.899 | 109.3 |

### Reversed: "RRF is worse than dense retrieval"

Phase 9 reported RRF at 0.669 against dense's 0.778 and concluded RRF lost. At
n=113, **RRF is 0.894 against dense's 0.876** — slightly better. That headline
was an artifact of twelve questions, exactly as the caveat predicted. It is
recorded here rather than quietly corrected in the earlier document.

### Held: the *mechanism* behind it is real

| Class | dense | lexical | hybrid (RRF) | weighted |
|---|---|---|---|---|
| paraphrase | **0.936** | 0.724 | **0.882** | **0.936** |
| version | 0.565 | 0.741 | 0.722 | **0.889** |
| lexical | 0.873 | 0.979 | 0.979 | 0.979 |
| distractor | 0.833 | 0.678 | 0.767 | 0.833 |

RRF still **damages paraphrase** relative to dense (0.882 vs 0.936), for the
reason Phase 9 identified: RRF uses only ranks, so BM25's rank-1 hit on a
paraphrase query counts as much as dense's, and on those queries BM25's hit is
noise. Weighted fusion normalises the scores and loses nothing.

What changed is that at n=113 the other classes compensate in the aggregate. The
per-class diagnosis was sound; the aggregate conclusion was not supported by the
sample. **This is the clearest case in the project for reporting mechanism
alongside aggregate.**

### Held: weighted fusion wins, and now by a real margin

0.938 against dense's 0.876 is **0.062 MRR ≈ 7 questions** at n=113, not the one
or two questions it was worth at n=12. The Phase 9 default change was right, and
it is now defensible.

### Held: the reranker does not pay for itself

0.867 — **below dense (0.876) and below both fusions** — for 109 ms, roughly 5×
the latency of weighted fusion. It is also *worse* than it looked at n=12, where
it tied dense. ADR-004's decision to keep it off by default stands and is
strengthened; the untested improvement remains reranking the dense list instead
of the fused one, so it stops inheriting the fusion's ordering.

### Held: lexical retrieval alone is not a retrieval system

0.812 overall, but paraphrase 0.724 against dense's 0.936 — and it is 200× faster
at 0.1 ms, which is why adding it costs nothing.

---

## 6. One honest weakness in the dataset

The generated paraphrase questions are **less pure than Phase 9's hand-written
ones**. To keep ground truth unambiguous across six sibling reports, each
generated paraphrase carries a qualifier:

```text
"Are customers paying slowly? (emea sales, Q2 FY2026)"
```

That parenthetical gives BM25 real terms to match, which is why `lexical` scores
0.724 on paraphrase here against **0.000** on Phase 9's hand-written set. The
class is therefore **easier than its name suggests**, and the dense-versus-BM25
gap on genuine paraphrase is understated.

Fixing it needs questions that are unambiguous without lexical overlap — which
means either a larger corpus or hand-written items. Recorded rather than left for
someone to discover in the numbers.

---

## 7. Running it

```bash
docker compose --profile core up -d qdrant
uv run python -m rag_service.corpus          # 12 docs
uv run python -m rag_service.evalset         # 113 eval items, hashed
uv run python -m rag_service.ingest

uv run llmops-eval run --mode hybrid_weighted     # full report + per class + ACL
uv run llmops-eval compare-modes                  # all five, the Phase 9 re-test
uv run llmops-eval baseline --mode hybrid_weighted # refuses a failing run
uv run llmops-eval gate                           # CI: exits non-zero on failure
uv run llmops-eval show-baseline
```

Every run is saved to `evaluations/<run_id>.json` with the configuration it
measured: dataset version and hash, index version, embedding model, reranker,
mode, `top_k`. A number without that attribution cannot be compared to anything.

**The baseline is a separate file**, not "the most recent run", so a regression
cannot silently become the new baseline. `baseline` refuses to record a run that
fails its gates, and a test asserts that invariant.

---

## 8. What is not evaluated yet

| Gap | Phase |
|---|---|
| **Generation metrics** — correctness, faithfulness, citation validity | 12 (judge), 15 (generation exists) |
| Abstention measured rather than declared | 15 |
| LLM-as-a-Judge, with its bias documented | 12 |
| Runs tracked in MLflow with parameters and artifacts | 13 |
| Regression gate on pull requests | 14 |
| `top_k` sweep against the 0.28 context precision | 14 |
| Human evaluation sample to calibrate the judge | 12 |
| Online / production evaluation | 22 |

Generation is the significant absence, and it is absent for a reason: there is no
generation in the system until Phase 15. Reporting a faithfulness score now would
mean inventing one.

---

## 9. Interview answer

> "Evaluation starts with the dataset, not the harness. Phase 9 compared five
> retrieval configurations on twelve questions and I recorded that one question
> was worth 0.083 MRR, so the differences were one or two questions and couldn't
> decide anything. Phase 11's first job was to make that statement false — 113
> items generated from the same seeded corpus, so ground truth can't drift,
> content-hashed so a metric is attributable to a dataset version, and weighted
> toward paraphrase and version disambiguation because those are the classes that
> actually discriminate between configurations.
>
> Re-running the comparison reversed my own headline. At n=12 I reported that RRF
> fusion was worse than dense retrieval; at n=113 it's slightly better. The
> mechanism I'd identified was real — RRF uses only ranks, so a weak BM25 hit
> counts as much as a strong dense one, and it still measurably damages paraphrase
> — but the aggregate conclusion wasn't supported by twelve questions. Three other
> conclusions held: weighted fusion wins, now by seven questions rather than one;
> the reranker costs 109 ms and scores *below* dense retrieval; and lexical search
> alone isn't a retrieval system.
>
> Metrics are arithmetic against planted ground truth, not a judge, because
> anything computable without a model should be. Unanswerable items are excluded
> from retrieval metrics and print n/a rather than zero — the metric is undefined
> there, not failed. And ACL is a hard gate measured across every identity: 678
> queries, zero unauthorized retrievals, and a run with perfect retrieval plus one
> leak still fails, which there's a test for."

---

## 10. Related

- [hybrid-retrieval.md](hybrid-retrieval.md) — the Phase 9 result this corrects
- [rag.md](rag.md) — the retrieval pipeline under evaluation
- [business-requirements.md](business-requirements.md) §4 — Q-1, Q-2, Q-6 thresholds
- [ADR-022](adr/ADR-022-evaluation-design.md) — harness design and alternatives
- [ADR-019](adr/ADR-019-authorization-model.md) — why ACL is measured per identity
- [versioning.md](versioning.md) — dataset as artifact class 8
