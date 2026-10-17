# LLM-as-a-Judge

**Status:** LOCAL. Code, prompts and tests complete; **results pending the first
run against real models** (§9). Nothing in this document reports a judge score
yet, because no judge has run.

---

## 1. What problem this solves

Phase 11 measured retrieval with arithmetic: the expected document is either in
the top 5 or it isn't. Generation has no such answer. "Is this answer faithful to
its sources?" has no closed form for a free-text answer, and a human cannot label
113 answers per change.

An LLM judge scales a human-like reading to every run. The cost of that is that
**the judge is a model with its own error**, and an evaluation that reports a
judge score without measuring the judge's error has replaced one unknown with
another.

So this phase builds two things, and the second matters more:

1. a judge harness: rubric prompts, schema-checked verdicts, recorded failures;
2. a calibration of the judge against answers whose right verdict is known.

---

## 2. Where it sits

```
                    OFFLINE (llmops-eval)                    ONLINE (product path)
dataset.json ──► retrieval pipeline ──► rag_answer@production ──► llm-application /chat
                 (in-process)            (prompt registry)            │ qwen2.5:3b
                                                                      ▼
                          answers-<ts>.jsonl  ◄── answer, context, citations,
                          (an artifact)            exact match, abstention
                                 │
          ┌──────────────────────┼─────────────────────────┐
          ▼                      ▼                         ▼
   judge (llama3.2:3b)    calibrate (probes)        human-sample (CSV, blind)
   judged-<ts>.jsonl      calibration-<ts>.json     human-agreement → kappa
```

Three decisions are visible in the diagram:

- **Answers are generated once and saved.** Judging the same file with a second
  judge, a new judge prompt or a human produces comparable numbers. Regenerating
  each time would mix generator variance into judge variance.
- **The judge calls its model directly**, not through llm-application. It is
  evaluation infrastructure: sharing the serving path would let a judge outage
  look like a product incident, and would make "judge = generator" the default.
- **The judge is a different model family from the generator** (Llama vs Qwen),
  to reduce self-preference bias (§5). The CLI warns when they are the same.

---

## 3. What is measured without a judge

Deterministic checks come first, and the judge is never asked a question that
has an exact answer. They are computed at generation time in
`llmops_eval.generation` and at request time in `rag_service.generation`:

| Metric | How | Why not the judge |
|---|---|---|
| Citation validity | every `[id]` must be a supplied source (set membership) | an exact question; a judge would add cost and error |
| Superseded-version citation | `DOC@v1` when only `v2` was supplied is invalid | same |
| Exact match | normalised reference figure is a substring of the answer | 12 planted figures are known exactly |
| Abstention | answer is `insufficient_evidence`, split answerable / unanswerable | a string comparison |

**Limit stated plainly:** only 17 of 113 items carry a reference answer (12
planted figures, 5 unanswerable). The rest were written to test retrieval and
carry an expected document, not an expected answer. Correctness is therefore
judged on 17 items; faithfulness, relevance and completeness are reference-free
and run on all 113.

---

## 4. The rubrics

All prompts are in `prompts/judge/`, released through the Phase 10 registry, at
the `dev` stage.

| Rubric | Inputs | Labels | Score |
|---|---|---|---|
| correctness | question, reference, answer | correct / partially_correct / incorrect | 1 / 0.5 / 0 |
| faithfulness | sources, question, answer | per claim: supported / unsupported / contradicted | fraction supported, **recomputed** |
| relevance | question, answer | relevant / partially_relevant / irrelevant | 1 / 0.5 / 0 |
| completeness | sources, question, answer | complete / partial / missing | 1 / 0.5 / 0 |
| pairwise | sources, question, answer A, answer B | A / B / tie | bias probes only |

Design rules, each with its reason:

- **Categorical labels, not 1–10.** A 3B model cannot hold a ten-point scale
  steady across items. Three labels with written definitions it can, and three
  labels map onto human labels for kappa.
- **Reason before verdict** in the JSON. The model writes its justification
  before committing to a label, rather than rationalising a label it already
  emitted.
- **Temperature 0, JSON mode.** Determinism is measured, not assumed (§6).
- **A failure is a row.** `parse_error`, `schema_error` and `transport_error` are
  recorded and reported as a failure rate beside each mean. Dropping them biases
  the mean towards the items the judge found easy.
- **Arithmetic is not delegated.** The faithfulness prompt asks the judge for its
  own fraction; the score used is computed from its claim list, and its own number
  is kept as `reported_score` so the gap is visible.
- **No claims is not perfect faithfulness.** An abstention makes no claims;
  scoring it 1.0 would reward saying nothing. It is recorded as `no_claims` with
  no score.

---

## 5. Known judge biases, and the probe for each

| Bias | What it is | Probe in this lab |
|---|---|---|
| **Self-preference** | a model rates text in its own style higher | judge family ≠ generator family; re-run `judge --judge-model qwen2.5:3b` on the same answers and compare |
| **Position** | in pairwise, preferring whichever answer comes first (or second) | gold vs wrong-figure asked in both orders; a flip is position, not quality |
| **Verbosity** | longer answers rated higher for the same content | `verbose_gold` = gold + filler must score as gold; pairwise gold vs verbose |
| **Leniency / error blindness** | missing small errors | `wrong_figure` changes **one digit**: crude errors would flatter the judge |
| **Rubric conflation** | marking a right answer wrong because it also says something unsupported | `fabricated_claim`: correctness should pass it, faithfulness should not |
| **Instability** | different verdicts on identical input | correctness asked `--repeats` times per probe at temperature 0 |
| **Capability ceiling** | a judge weaker than the task | a 3B judge on financial figures; measured by error recall, not assumed |

---

## 6. Calibration: the probes

`llmops-eval calibrate <answers.jsonl>` builds probes from the generation run's
real contexts, so the judge sees exactly what a generated answer would have seen:

| Variant | Answer | Expected correctness | Expected faithfulness |
|---|---|---|---|
| gold | reference figure + correct citation | correct | faithful |
| wrong_figure | one digit changed | incorrect | unfaithful |
| fabricated_claim | gold + invented cause | correct | unfaithful |
| verbose_gold | gold + filler, no new fact | correct | faithful |
| false_abstention | `insufficient_evidence` on answerable | incorrect | — |
| gold_abstention | `insufficient_evidence` on unanswerable | correct | — |
| hallucinated | a figure on unanswerable | incorrect | unfaithful |

Reported, framed as a classifier where "positive" means the judge flags an error:

- **error recall**: planted errors caught / planted errors;
- **false-alarm rate**: correct answers flagged / correct answers;
- per-variant accuracy for both rubrics, instability count;
- position outcome (consistent-correct / consistent-wrong / flip);
- verbosity outcome (prefers concise / prefers verbose / tie).

**THIS IS A SIMPLIFICATION.** Perturbations are synthetic and cover failure
types I thought of. A judge can pass every probe and still miss the errors a
real generator makes. That gap is what the human sample (§7) is for, and
`agreement_with_exact_match` checks the judge on real generated answers: its
disagreements are listed for a human, not counted as judge errors, because exact
match is itself wrong on a correctly reformatted figure (`€959k`).

---

## 7. Human evaluation

`human-sample` writes 30 answers, stratified by hardness class so the rare
classes are present, to a CSV with **no judge output in it**. A labeller shown the
judge's verdict agrees with it more than the answer deserves (anchoring).

Labels: `human_faithful` (supported / unsupported) and `human_relevance`
(relevant / partially_relevant / irrelevant). `human-agreement` reports raw
agreement **and Cohen's kappa**:

> If 90% of answers are correct, a judge that always says "correct" agrees with a
> human 90% of the time and has kappa 0. Raw agreement on a skewed set measures
> the skew. A unit test pins exactly this case.

One labeller is a weak gold standard: in production, two labellers and their
inter-annotator kappa set the ceiling the judge can be expected to reach.

---

## 8. Failure modes

| Failure | Detection | Effect if missed |
|---|---|---|
| Judge model down | `transport_error` count | none silently; the run reports it |
| Judge emits prose / wrong label | `parse_error` / `schema_error` rate | mean over survivors only, inflated |
| Judge prompt edited in place | registry hash check at load | scores attributed to a prompt that no longer exists |
| Judge = generator | CLI warning; header records both models | self-preference inflates scores |
| Context too long for a 3B judge | faithfulness `no_claims` / errors on long contexts | faithfulness silently measured on a truncated reading |
| Judge drift after a model update | judge model recorded per run; re-run calibration | yesterday's and today's scores not comparable |

---

## 9. Results

**Pending.** Requires Ollama with `qwen2.5:3b` and `llama3.2:3b`, the `core` and
`rag` Compose profiles, and `LLMOPS_LLM__PROVIDER=ollama`. Commands, in order:

```powershell
uv run llmops-eval generate                       # ~113 generations
uv run llmops-eval judge evaluations/answers/answers-<ts>.jsonl
uv run llmops-eval calibrate evaluations/answers/answers-<ts>.jsonl --judged evaluations/judged/judged-<ts>.jsonl
uv run llmops-eval judge evaluations/answers/answers-<ts>.jsonl --judge-model qwen2.5:3b   # self-preference
uv run llmops-eval human-sample evaluations/answers/answers-<ts>.jsonl
# label the CSV, then
uv run llmops-eval human-agreement evaluations/human/<file>.csv evaluations/judged/judged-<ts>.jsonl
```

What I expect, written before the run so it can be wrong: a 3B judge catches
`false_abstention` and `hallucinated` reliably, misses a meaningful share of
one-digit `wrong_figure` errors, and passes `fabricated_claim` on faithfulness
more often than it should. Context precision of 0.28 (Phase 11) means ~72% of
each context is irrelevant, so faithfulness judgements on long contexts are the
least trustworthy number here.

---

## 10. Cost and scaling

Per full run: 113 generations + about 113 × 3.15 judge calls (correctness on 17
only) + calibration ≈ 70 probes × (2 correctness + ~0.7 faithfulness) + ~36
pairwise calls. Locally the cost is GPU time; with an external judge at
~1.5k input tokens per call it is ~0.9M tokens per run. That is why the judge
runs on saved answers and on change, not on every request. Online evaluation
(Phase 22) samples production traffic rather than judging all of it.

---

## 11. Interview answer

> "I treat an LLM judge as a measuring instrument, so it gets calibrated before
> its numbers are used. Anything with an exact answer never goes to the judge:
> citation validity is set membership against the sources that were supplied,
> and planted figures are exact match. The judge only gets the open questions:
> faithfulness, relevance, completeness. It returns categorical labels, reason
> first, as schema-checked JSON. A parse failure is a recorded row with a failure
> rate beside the mean, because dropping failures inflates the mean.
>
> Then I score the judge itself on answers whose verdict I know by construction.
> A one-digit wrong figure tests whether it notices small errors. A correct
> answer with a fabricated cause tests whether it keeps correctness and
> faithfulness apart. Gold padded with filler tests verbosity bias. A pairwise
> comparison asked in both orders tests position bias. The judge is a different
> model family from the generator, against self-preference. And I compare it with
> blind human labels using kappa, not raw agreement, because on a skewed set a
> judge that always says 'correct' can agree 90% of the time and still have a
> kappa of zero. So the judge is a scalable proxy for a human reading, with a
> measured error bar. It isn't ground truth."
