# ADR-022: Evaluation — generated ground truth, arithmetic metrics, a library not a service

**Status:** Accepted
**Date:** 2026-10-04
**Phase:** 11

## Context
Phase 9 compared five retrieval configurations on 12 hand-written questions and
recorded that one question was worth 0.083 MRR, so differences below ~0.15 were
indistinguishable from noise. Re-running the same comparison at n=113 reversed
the headline conclusion: RRF went from 0.669 (apparently worse than dense) to
0.894 (slightly better). The sample size, not the retrieval strategy, produced
that result.

So the design problem is not "how do we compute Recall@k". It is: how does an
evaluation produce numbers that can safely decide something.

## Decision

**1. Ground truth is generated from the corpus, not written by hand.**
`rag_service.evalset` derives 113 items from the same seeded generator that
produced the documents.

**2. Metrics are arithmetic, not judged.** Recall@k, MRR, nDCG@5, context
precision and recall are computed against known `(doc_id, version)` locations.
No model opinion is involved in any retrieval metric.

**3. The dataset is a hashed artifact.** `dataset.json` carries a version and a
content hash (`36b90db3c48d`), and every run records both. A metric without its
dataset version is not comparable to anything (ADR-015 class 8).

**4. ACL is a hard gate, measured across every identity.** Any unauthorized
retrieval fails the run regardless of every other number. 6 identities × 113
questions = 678 queries per run.

**5. Unanswerable items are excluded from retrieval metrics** and displayed as
`n/a` rather than `0.000`. There is nothing to recall; a zero reads as a failure
when the metric is undefined.

**6. A library and CLI, not a service.** `packages/llmops-eval` with
`llmops-eval` as an entry point. This is a **deviation from
`docs/architecture.md`** layer 13, which lists `apps/evaluation-service`.

**7. The baseline is a separate file that refuses a failing run.**
`evaluations/baseline.json`, not "the most recent run", so a regression cannot
silently become the new baseline.

## Alternatives considered

| Option | Why not |
| --- | --- |
| **Hand-written questions only** | what Phase 9 did. Higher quality per item — the hand-written paraphrases had *zero* lexical overlap, which generated ones cannot guarantee (see Consequences) — but 12 items took real effort and 113 would not have been written. The real set keeps all 12 hand-written items alongside the generated ones |
| **LLM-generated questions** | scales further and reads more naturally. Rejected for Phase 11 because the generator would share a model family with the system under test, and because ground truth would then need verification — a dataset whose labels are themselves model output is a weaker instrument than one derived from a seeded generator. Worth revisiting in Phase 12 where judge bias is already being documented |
| **RAGAS / promptfoo / DeepEval** | real frameworks with these metrics implemented. Rejected *for the retrieval metrics* because they are ten lines of arithmetic and implementing them is the learning objective; promptfoo is still the Phase 14 choice for regression gating, where a framework earns its place |
| **LLM-as-a-Judge for retrieval** | a judge for something computable is strictly worse: it costs money, adds bias, and cannot be unit-tested. Judges are for generation quality (Phase 12), where no arithmetic answer exists |
| **`apps/evaluation-service` now** | offline evaluation is a batch job. A service would need a scheduler, an API and a queue to do what a CLI does in one command, and nothing calls it yet. The service arrives in Phase 22 when *online* evaluation needs an endpoint |
| **MLflow as the only store** | Phase 13's job. JSON files land first so runs exist before a tracking server does, and so a run is readable without one |
| **Latest run as baseline** | a regression then becomes the baseline on the next run, which silently raises the floor and hides the regression |
| **One identity for the ACL check** | cannot demonstrate "no identity ever retrieves above its clearance", which is the property the gate claims |

## Trade-offs

+ 113 items means one question is 0.0088 MRR, against 0.0833 at n=12 — enough to
  resolve a 0.06 difference
+ ground truth cannot drift from the corpus, because one generator produces both
+ the whole harness runs offline, with no API key and no model call
+ metrics are unit-tested, so a subtly wrong metric cannot quietly invalidate
  every conclusion drawn from it
+ the same dataset measures quality *and* ACL, rather than needing two

− **generated paraphrases are less pure than hand-written ones.** Each carries a
  disambiguating qualifier (`"(emea sales, Q2 FY2026)"`) so ground truth is
  unambiguous across six sibling reports — and that parenthetical gives BM25
  real terms. BM25 scores 0.724 on paraphrase here against 0.000 on Phase 9's
  hand-written set, so the class is easier than its name suggests and the
  dense-versus-lexical gap is understated. Recorded in `docs/evaluation.md` §6
− a synthetic corpus bounds what any of this can prove. Recall@5 is 1.000 in
  every class, which says more about 18 chunks than about retrieval
− no generation metrics at all, because there is no generation until Phase 15.
  Faithfulness and citation validity are declared and empty rather than invented
− abstention is recorded as unmeasured; retrieval cannot abstain
− deviating from the architecture document needs the architecture document
  updated, which is a cost paid in Phase 40's review

## Consequences

- `evaluations/` holds one JSON per run plus `baseline.json`, each recording
  dataset version and hash, index version, embedding model, reranker, mode and
  `top_k`.
- `llmops-eval gate` is the CI entry point (Phase 34) and already fails on
  Q-1, Q-2 or Q-6.
- Phase 12 adds judged generation metrics *alongside* these, never replacing
  them, and must document judge bias and calibrate against a human sample.
- Phase 13 pushes these runs into MLflow; the JSON files remain the local record.
- Phase 14 adds the pull-request regression gate and the `top_k` sweep against
  the measured 0.28 context precision.
- `docs/architecture.md` layer 13 should be amended to say *library plus CLI for
  offline evaluation, service for online*. Flagged for Phase 40.
- Any future claim from this harness must cite the dataset hash. The Phase 9
  reversal is the standing reminder of why.
