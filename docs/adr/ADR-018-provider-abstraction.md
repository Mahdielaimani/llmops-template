# ADR-018: One provider interface for mock, local and external LLMs

**Status:** Accepted
**Date:** 2026-09-26
**Phase:** 2

## Context
The platform must run in three modes (`docs/environment.md` §6): a deterministic
mock for tests and CI, a local vLLM or Ollama server on the 8 GB GPU, and an
external API when a stronger model is worth paying for. Callers — `/chat` now,
`/rag` in Phase 8, the agent loop in Phase 24 — must not branch on which one is
active, or every feature gets written three times and the mock silently drifts
from the real thing.

## Decision
A single `LLMProvider` protocol in
`apps/llm-application/providers/base.py` with two methods, `complete()` and
`stream()`, plus `aclose()`. Two implementations:

- **`MockProvider`** — deterministic, with configurable `prefill_s` and
  `per_token_s` so the *shape* of inference (a prefill pause, then per-token
  gaps) is reproduced. Streaming, timeout and backpressure paths get exercised
  without a GPU.
- **`OpenAICompatProvider`** — one HTTP client for vLLM, Ollama and OpenAI,
  since all three speak `/v1/chat/completions`.

Selected by `build_provider(settings)` from `LLMOPS_LLM__PROVIDER`. Instantiated
in `lifespan`, never at import, so importing a module never opens a socket.

Four decisions inside the interface that carry weight:

1. **Unknown usage is `None`, never estimated.** A guessed token count silently
   corrupts the cost model in `docs/metrics-and-capacity.md` §6 and §7. The
   non-streaming path also reports `ttft_ms = None`, because a non-streaming
   response carries no first-token signal — reporting total latency as TTFT
   would be a plausible and wrong number.
2. **TTFT is timed at the first *content* token.** An OpenAI-compatible server
   opens with a role-only delta carrying no text; timing from it reports a token
   that was never generated. There is a test for exactly this.
3. **No retries.** `CLAUDE.md` §55: a retried LLM request duplicates GPU work and
   tokens, so one timeout becomes 3× load on an already-saturated engine. Retry
   and fallback belong in the model gateway (Phase 15), which can see queue depth.
4. **Upstream bodies are logged, never returned.** A provider error body can echo
   the prompt back; the caller gets our envelope and a code.

## Alternatives considered
| Option | Why not |
| --- | --- |
| LangChain / LiteLLM as the abstraction | another dependency to learn and debug for the one thing we need; hides TTFT/ITL measurement, which is the point of this phase; LiteLLM is a reasonable choice for the Phase 15 gateway and is revisited there |
| Official `openai` SDK | works, but pulls a provider-branded client for a generic protocol, and its streaming types would leak into ours |
| Build the model gateway now (Phase 15 early) | the gateway's value is routing, fallback and cost across providers — none of which exist yet with one provider and no cost data |
| Skip the mock, always hit a real model | CI would need a GPU; tests would be non-deterministic; a saturated GPU would block unrelated work |
| A separate `packages/llmops-llm` package | no second caller yet. It moves to a package when the model gateway needs it (ADR-011: no abstraction before the second caller) |

## Trade-offs
+ one interface, three modes, zero branching in callers
+ TTFT / TPOT / ITL are measured by *us*, which is the phase's learning objective
+ the mock reproduces inference shape, so timing code is testable at millisecond scale
− `OpenAICompatProvider` assumes OpenAI-shaped responses; Anthropic's Messages API
  is not, and raises `NotImplementedError` until the Phase 15 gateway
− a protocol with two implementations is borderline premature; justified because
  both exist today and the third mode (external) is the same class
− error mapping is coarse (429 → rate-limited, 408/504 → timeout, else
  unavailable); finer mapping arrives with the gateway's circuit breaker

## Consequences
- `/rag` (Phase 8) and the agent loop (Phase 24) consume the same interface.
- Phase 6 needs no application change: point `LLMOPS_LLM__BASE_URL` at the vLLM
  container and the provider switches.
- Phase 7 (streaming metrics) has nothing left to build for `/chat` — the
  measurement is already in `Timings`.
- Phase 15 moves this package to `packages/`, adds retry, fallback and cost, and
  re-evaluates LiteLLM as the gateway's internals.
- The mock's defaults are zero-delay, so timings are degenerate unless a test sets
  them. Tests that assert on timing must configure `prefill_s` / `per_token_s`.
