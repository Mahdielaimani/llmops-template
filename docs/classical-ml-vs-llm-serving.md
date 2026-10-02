# Classical ML Serving vs LLM Serving

**Status:** LOCAL / measured 2026-10-01 on the audited machine (`docs/environment.md`).
Phase 3. Every number below was measured, not asserted; the measurement method and
its one large confound are stated in §1.

Two services, same machine, same moment:

| | `classical-ml` :8090 | `llm-application` :8080 |
| --- | --- | --- |
| Task | will this invoice be paid late? | free-text chat |
| Model | LogisticRegression + StandardScaler | mock provider (Phase 6 swaps in vLLM) |
| Artifact | **997 bytes** | 4–15 GB of weights |

---

## 1. Measurement method, and the 43 ms trap

The first benchmark run reported ~44 ms p50 for **all three** endpoints —
single prediction, batch of 100, and mock chat. Identical numbers across
workloads that differ by orders of magnitude is a measurement artefact, not a
result.

| Measured from | classical `/predict` p50 | llm `/chat` p50 |
| --- | --- | --- |
| Windows host (`localhost:8090`) | 43.99 ms | 43.99 ms |
| Inside the Compose network (`classical-ml:8090`) | **1.26 ms** | **1.03 ms** |
| Inside the handler (`latency_ms` in the response) | **0.26 ms** | 0.01 ms |

**~43 ms of every host-side request is Docker Desktop's Windows→WSL2 port
proxy.** It is constant, it is the same for every endpoint, and it would have
swamped every latency conclusion in this document.

Consequences adopted for the rest of the lab:
- benchmarks run **inside the Compose network**, never from the Windows host
- every handler reports its own `latency_ms`, so transport can be subtracted
- Phase 30 load tests run Locust as a container on the same network, or the
  bottleneck "found" will be the port proxy

This is the Phase 0 core principle arriving early: measure, then ask what the
number actually contains.

---

## 2. Measured results

All figures from inside the Compose network, 200 sequential requests after a
warm-up call, mock LLM provider.

| | p50 | p95 | server-side p50 | response bytes |
| --- | --- | --- | --- | --- |
| `/predict` (1 invoice) | 1.261 ms | 1.877 ms | 0.262 ms | 182 |
| `/predict/batch` (100 invoices) | 2.219 ms | 3.036 ms | 0.442 ms | 18 548 |
| `/chat` (mock, 32 max tokens) | 1.029 ms | 1.580 ms | 0.010 ms | 289 |

**The batch row is the headline.** 100 predictions cost 0.442 ms against 0.262 ms
for one — a 100× increase in work for a **1.7× increase in time**. One
`predict_proba` call is a single matrix multiply over a (100, 6) array; the
per-request cost is almost entirely framework overhead, not arithmetic.

An LLM cannot do this. Doubling the batch in an LLM engine roughly holds
per-token latency only while KV cache allows it, then degrades, then OOMs
(`docs/metrics-and-capacity.md` §5: ~11 concurrent sequences at 4k context on
this GPU). Classical batching is free; LLM batching is a scheduling problem
bounded by VRAM.

**`/chat` at 0.01 ms server-side is not a result.** It is the mock provider doing
no work — framework overhead with inference removed. It is a floor, and the
honest comparison arrives in Phase 6. What it does establish: ~1 ms of the
round-trip is FastAPI + network, so anything above that in Phase 6 is the model.

---

## 3. Startup and artifact size

```text
{"event":"startup","model_version":"1.0.0","artifact_bytes":997,"model_load_ms":1115.83}
```

1.1 seconds to load a 997-byte artifact. The artifact read is microseconds; the
time is **importing scikit-learn, scipy and numpy**. Two things follow:

- Load time is a property of the *runtime*, not the model size.
- The image is **607 MB** to serve a 997-byte model, against 251 MB for
  `llm-application`. 356 MB of that is numpy + scipy + sklearn. The
  model-to-runtime ratio is roughly 1 : 600 000.

Against vLLM (Phase 6): engine init was measured upstream at 8.2 s on an H200
after the v0.30.0 improvements, and 1–3 minutes is normal on consumer hardware
with a 7B model. That gap is exactly why ADR-013 splits liveness from readiness
and why Kubernetes needs a `startupProbe` for the LLM service and not for this
one. `classical-ml` is ready in a second; a probe with a 5-second start period is
sufficient.

---

## 4. The operational differences that actually matter

| | Classical ML | LLM |
| --- | --- | --- |
| **Input** | fixed 6-field schema, typed | free text, no schema |
| **Bad input** | `422 validation_error` | answered anyway, plausibly |
| **Output** | one float | variable-length token sequence |
| **Determinism** | identical input → **identical output, always** | sampled; temperature, seed, engine version all move it |
| **Compute profile** | CPU, ~0.3 ms, predictable | GPU; prefill compute-bound, decode **memory-bandwidth**-bound |
| **Memory** | constant | weights + KV cache growing with context × concurrency |
| **Batching** | one matrix multiply, near-free | engine scheduler, continuous batching, VRAM-bounded |
| **Streaming** | meaningless — one value | essential; TTFT is a separate SLO from total latency |
| **Metrics that matter** | p50/p95, accuracy vs baseline | TTFT, ITL, TPOT, tokens/s, queue time, KV utilisation |
| **Scaling knob** | add replicas (stateless, cheap) | `max_num_seqs`, context length, quantization — **replicas do not help** |
| **Cost driver** | requests/second | input tokens × output tokens × concurrency |
| **Artifact** | 997 B, versioned file | 4–15 GB, downloaded, quantized, revision-pinned |
| **Rollback** | swap a pointer file, instant | reload weights, minutes, VRAM must be free |
| **Failure mode** | wrong number, measurable against ground truth | **wrong answer that reads as confident** |

### Determinism, demonstrated

`tests/unit/test_classical_ml.py::test_prediction_is_deterministic` asserts that
six identical calls return a byte-identical probability. There is no equivalent
test possible for `/chat`, and that single fact drives most of the rest of this
platform: because LLM output is not reproducible, quality has to be measured
statistically over a dataset (Phase 11), regression-tested against a baseline
rather than a fixture (Phase 14), and judged (Phase 12) rather than asserted.

### Schema, demonstrated

```text
$ curl -X POST :8090/predict -d '{"amount_eur":"twenty thousand", ...}'
{"error":{"code":"validation_error", ...
          "msg":"Input should be a valid number, unable to parse string as a number"}}
```

The classical service **cannot** be asked a malformed question. An LLM given
"amount: twenty thousand" produces an answer, and nothing in the transport
layer can tell that it was nonsense. Validation moves from the edge (free) to
output guardrails and evaluation (expensive) — Phases 28 and 11.

---

## 5. Model quality, reported honestly

From `models/invoice-late-payment/model-1.0.0.json`:

| Metric | Value |
| --- | --- |
| ROC AUC | 0.7538 |
| Accuracy | 0.7652 |
| **Majority-class baseline accuracy** | **0.7263** |
| Precision | 0.6696 |
| Recall | 0.2813 |
| Positive rate | 0.2737 |

Three deliberate disclosures:

1. **Accuracy alone would be a lie.** 0.7652 looks adequate until you see that
   always predicting "on time" scores 0.7263. The artifact carries both numbers
   so the comparison cannot be omitted. An LLM eval report has the same hazard:
   a correctness score without a baseline is decoration.

2. **Recall is 0.2813 at threshold 0.5.** The model flags fewer than a third of
   genuinely late invoices. 0.5 is a default, not an operating point — choosing
   one requires the business cost of a missed late payment versus a false alarm,
   which this lab does not have. Recorded in `model.py` rather than quietly
   tuned away.

3. **The first attempt scored AUC 0.61** — worse than useless on a 73% majority
   class. The synthetic generator was fixed (terms centred so each contributed
   variance instead of a near-constant offset, noise reduced 8% → 5%), not the
   evaluation. Tuning a generator until the model flatters the demo is the
   data-science equivalent of deleting a failing test.

Also observed: the model returns `probability: 1.0` on extreme inputs
(€480 000, 7 days, new customer, no PO, 0.8 prior late ratio). Logistic
regression saturates and is over-confident at the tails — calibration is a real
concern that this lab does not address.

---

## 6. What transfers to LLM serving, and what does not

**Transfers:**
- the artifact is versioned, content-hashed and carries its own metadata
  (ADR-015 artifact class 3 — the same discipline as prompts and indexes)
- a pointer file (`current.txt`) selects the live version, so rollback is a
  pointer swap; this is the small-scale form of the Qdrant alias flip
- the model is loaded once in `lifespan`, never at import
- liveness and readiness are separate, from the same `llmops-core` registry
- one error envelope, one log shape, `X-Request-ID` propagation — identical code

**Does not transfer:**
- "latency" as a single number. An LLM needs TTFT, ITL, TPOT and total, because
  a user perceives the first token, not the last.
- unit tests as a correctness gate. Non-determinism forces statistical
  evaluation over a dataset.
- "add replicas" as the scaling answer. Replicas multiply requests arriving at
  one saturated GPU (Phase 31 proves this with numbers).
- free batching. LLM batching is a VRAM-bounded scheduling decision.
- constant memory. KV cache growth is the concurrency ceiling.

---

## 7. Reproduce it

```bash
cd C:\llmops-project
uv run python -m classical_ml.train          # writes models/invoice-late-payment/
uv run poe check                             # 48 unit tests
docker compose --profile core --profile classical up -d --build

# the trap: from the Windows host, every endpoint reads ~44 ms
uv run python benchmarks/compare_serving.py

# the truth: from inside the Compose network
docker compose exec -T api python -c "import httpx,time,statistics; ..."

curl -s :8090/model | jq                     # artifact provenance + metrics
docker compose logs classical-ml | grep startup
```

---

## 8. Interview answer

> "Classical serving is a fixed-schema function call: six floats in, one float
> out, deterministic, about 0.3 ms on CPU, and a batch of a hundred costs 1.7×
> a single call because it is one matrix multiply. LLM serving breaks every one
> of those assumptions — variable-length input and output, autoregressive
> generation so you need TTFT separately from total latency, memory-bandwidth
> bound decode rather than compute-bound, a KV cache that grows with context
> times concurrency and sets a hard concurrency ceiling, and non-deterministic
> output so correctness becomes statistical rather than a unit test. The
> operational consequence people miss is that replicas fix a classical
> bottleneck and do nothing for a saturated GPU. And the first thing I measured
> here turned out to be Docker Desktop's port proxy — 43 ms on every request,
> identical across endpoints — which is why benchmarks run inside the network
> and every handler reports its own latency."
