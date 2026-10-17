# ADR-023: LLM-as-a-Judge — offline, calibrated, a different model family

**Status:** Accepted
**Date:** 2026-10-17
**Phase:** 12

## Context
Phase 11 measures retrieval arithmetically. Generation quality (faithfulness,
relevance, completeness) has no closed-form metric, and only 17 of 113 dataset
items carry a reference answer. Generation itself did not exist until this
phase: `/rag` returned `answer: null`.

Hardware: one 8 GB GPU, and vLLM is deferred (Phase 6 ⊘, disk). An external API
judge was offered and declined in favour of local models.

## Decision
1. **Generation in `/rag`** calls llm-application `/chat` by Compose service name
   with `rag_answer` resolved from the registry at startup. Citation validity is
   checked in code, not by a judge.
2. **Answers are an artifact.** `llmops-eval generate` saves answers with their
   contexts; `judge`, `calibrate` and `human-sample` all consume that file.
3. **Judge = `llama3.2:3b` on Ollama; generator = `qwen2.5:3b` on Ollama.** Different
   families against self-preference. Both 3B so they co-reside in 8 GB VRAM.
4. **The judge calls its model directly**, not via llm-application.
5. **Categorical labels, reason first, JSON mode, temperature 0.** Failures are
   recorded statuses with a reported rate. Faithfulness is recomputed from claims.
6. **Calibration is part of the phase**, not an afterthought: synthetic probes with
   known labels, position and verbosity probes, and a blind human sample scored by
   Cohen's kappa.

## Alternatives
- **External frontier judge (OpenAI/Anthropic).** Stronger, and a different vendor
  than the generator. Rejected for now (declined; no key). The judge URL and model
  are flags, so it is a configuration change, not a code change.
- **Same local model as judge.** Zero extra VRAM, but self-preference is built in.
- **Judge inside the request path (online, every request).** Doubles model cost per
  request and couples product latency to the judge. Deferred to Phase 22 as sampling.
- **Ragas / DeepEval.** Mature metric suites. Rejected as the core because their
  scores hide the prompt and the parse-failure handling, which are exactly what this
  phase is meant to make visible. Can be added later as a cross-check.
- **1–10 numeric scores.** Finer-grained in theory; unstable on a 3B model and not
  comparable with human labels.

## Trade-offs
- A 3B judge has a capability ceiling, which is why its error is measured
  rather than assumed.
- Synthetic probes only cover failure types someone thought of; the human sample
  covers the rest, but n=30 with one labeller is a weak gold standard.
- Saving answers costs disk and means a judge run can be stale relative to code;
  the answers header records prompt, model, index and dataset versions for that reason.

## Consequences
- Every judge number is attributable: judge model, judge prompt version, generator
  model, generator prompt, dataset hash.
- The Phase 14 regression gate can use judge scores only together with their
  calibration (error recall, false-alarm rate) and failure rate.
- Swapping in an external judge later is one flag and one re-calibration.
