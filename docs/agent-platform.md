# Agent Platform — A2A, Coordination, Skills, Trust, Human Interrupt

**Status:** design (Phases 24–29b). Nothing here is production-ready.

Covers the five questions that separate "an agent demo" from an agent
platform: how agents talk to each other, how work is decomposed and
joined, what a skill is, how an agent's output is *verified* rather than
trusted, and how a human interrupts a risky action.

---

## 1. Two protocols, two jobs

| | **MCP** (Phase 26) | **A2A** (Phase 26b) |
| --- | --- | --- |
| Connects | agent → tool / resource | agent → **agent** |
| Other side is | a passive function | an autonomous peer with its own model, context, tools |
| Unit of work | tool call (request/response) | **task** with a lifecycle |
| Discovery | tool list / schema | **Agent Card** at `/.well-known/agent-card.json` |
| Duration | short | long-running, resumable, streamable |
| Transport | stdio / HTTP | JSON-RPC or HTTP + SSE, optional push notification |

Both are needed and they are not substitutes: MCP gives an agent *hands*,
A2A gives it *colleagues*.

**Neither is a security boundary.** An MCP server's tool list is a
capability catalogue, not an authorization decision; an A2A peer's reply
is untrusted input that may itself be attacker-influenced. Authorization
and output sanitization stay in our application (§5, and `CLAUDE.md` §62).

### A2A task lifecycle

```text
submitted → working → ┬→ completed
                      ├→ input-required   ← human approval gate lives here (§6)
                      ├→ failed
                      └→ canceled         ← kill switch
```

Task state is persisted in **Postgres**, not in process memory: a
supervisor restart must not lose a half-finished delegation, and an
approval may arrive minutes later.

### Agent Card (what we publish per agent)

```json
{
  "name": "financial-analysis-agent",
  "version": "1.3.0",
  "description": "Variance and ratio analysis over retrieved financial statements",
  "url": "http://agent-financial:8080/a2a",
  "capabilities": { "streaming": true, "pushNotifications": false },
  "authentication": { "schemes": ["bearer"] },
  "skills": [
    { "id": "variance_analysis", "version": "2.1.0",
      "description": "Compare actuals vs plan across periods",
      "inputModes": ["text"], "outputModes": ["text", "data"] }
  ]
}
```

The card is generated from the agent registry (§3) — never hand-written,
so what is advertised is what is deployed.

---

## 2. Coordination: how agents talk and wait

Three patterns, chosen per task by the supervisor:

```text
SEQUENTIAL      supervisor → A → B → C → answer
                simple, debuggable; latency = Σ, tokens = Σ

FAN-OUT / JOIN  supervisor ─┬→ research-agent      ┐
                            ├→ financial-agent     ├→ await → merge → answer
                            └→ compliance-agent    ┘
                latency ≈ max(branches) + merge; tokens = Σ (cost is additive!)

PIPELINE        research → analysis → writer      (output_i = input_{i+1})
                each stage independently versioned and evaluated
```

**"One waits for the others" = the join**, and a join needs four explicit
decisions — defaults that silently differ are how multi-agent systems hang:

| Decision | Options | Our default |
| --- | --- | --- |
| Completion policy | `all` / `first_success` / `quorum(k)` / `best_effort` | `all` with deadline |
| Per-branch deadline | absolute vs shared budget | shared wall-clock budget, per-branch cap |
| Partial failure | fail whole task / degrade with note / retry branch | **degrade + state which branch failed in the answer** |
| Budget | per-branch vs shared | **shared** token/cost budget, debited atomically |

Implementation shape (agent-service):

```python
async def join(branches, *, deadline_s, policy="all"):
    tasks = [asyncio.create_task(run_branch(b)) for b in branches]
    done, pending = await asyncio.wait(tasks, timeout=deadline_s)
    for t in pending:
        t.cancel()                       # and mark the A2A task canceled
    results = [t.result() for t in done if not t.exception()]
    ...
```

Shared state between agents uses a **blackboard** in Redis keyed by
`task_id` (intermediate findings, retrieved chunk ids, running cost), so
branches do not re-retrieve the same documents and the supervisor can
inspect progress mid-run.

Hard limits, enforced by the runtime not the prompt:
`max_delegation_depth=2`, `max_branches=4`, `max_iterations` per agent,
wall-clock deadline, shared token and cost budget, per-agent tool allowlist.

---

## 3. Skills — the versioned unit of capability

A **skill** is not a Python function. It is a named, versioned, evaluated
capability that an agent advertises and a supervisor routes to.

```yaml
# skills/variance_analysis/skill.yaml
id: variance_analysis
version: 2.1.0
owner: platform-team
description: Compare actual vs planned figures across periods and explain drivers.
inputs:  {schema: schemas/variance_in.json}   # entity, periods, metric
outputs: {schema: schemas/variance_out.json}  # rows + narrative + citations
prompt:  {ref: prompts/agents/variance_analysis@2.1.0}
tools:   [document_search, calculator, table_extract]   # allowlist
limits:  {max_iterations: 6, max_tokens: 8000, timeout_s: 90, risk_tier: compute}
eval:    {dataset: evaluations/skills/variance_analysis@1.2.0, thresholds: {correctness: 0.8, citation_validity: 0.9}}
```

Why skills exist:
- **Routing** — the supervisor picks a skill by description/schema, not a
  hard-coded agent name; adding an agent does not change the supervisor.
- **Discovery** — the Agent Card's `skills` array is generated from these files.
- **Evaluation** — each skill owns an eval set, so quality is measured per
  capability rather than per whole system (Phase 11/14 gates apply per skill).
- **Governance** — the skill is what gets promoted, rolled back, and
  attributed in the release manifest (`docs/versioning.md`).

Initial skill set for this domain: `document_search`, `table_extract`,
`financial_calc`, `variance_analysis`, `citation_check`, `compliance_review`,
`report_write`.

---

## 4. Trust and verification — never trust the agent

An agent's self-report ("I checked the sources") is generated text, not
evidence. Four independent layers, strongest first:

### L1 — Deterministic verification (code, not a model)
The only layer that cannot be talked out of its answer.
- every cited `chunk_id` exists, is in the user's ACL, and **contains the
  claimed figure** (string/number match against the chunk text)
- arithmetic recomputed in Python from extracted table cells
- output validates against the skill's JSON schema
- every numeric claim in the answer maps to ≥1 citation; uncited numbers
  are a hard fail
- tool calls actually made ⊆ skill's allowlist

### L2 — Independent verifier agent
Different model and prompt, sees `{question, answer, retrieved sources}`
but **not** the producer's reasoning trace (so it cannot be primed by it).
Returns per-claim `supported | unsupported | contradicted` + severity.
Disagreement between L1 and L2 is itself a signal worth alerting on.

### L3 — Trajectory evaluation (process, not just outcome)
Right answer by luck is still a defect. Judge the run:
did it retrieve before asserting? were the tools appropriate? did it stop
when evidence was sufficient? iteration count within norm? Measured
offline on recorded traces (Phase 14) and sampled online.

### L4 — Attestation and audit
Every answer carries a signed record: `release_manifest`, prompt/model/
index versions, document ids + versions used, tools executed, approvals
granted, cost. Reproducible after the fact — the auditor requirement from
`docs/business-requirements.md`.

### Abstention is a feature
The agent must be able to return `insufficient_evidence` instead of
guessing. Measured as **coverage vs accuracy**: an agent answering 70 % of
questions at 95 % correctness beats one answering 100 % at 75 %. Tracked
as a first-class metric, and forcing an answer is treated as a regression.

---

## 5. Where security is enforced (six points, none of them the model)

| # | Point | Enforces | Phase |
| --- | --- | --- | --- |
| 1 | Kong (edge) | authn, RBAC group, rate limit, body size | 16 |
| 2 | Retrieval filter | ACL — the model never sees a disallowed chunk | 11 |
| 3 | Tool broker | allowlist, per-user authz, param schema, resource caps | 24 |
| 4 | Peer (A2A) authz | which agent may call which, with which scopes; remote output treated as untrusted data | 26b |
| 5 | Output guardrails | schema, citation validity, PII/secret scan, unsupported claims | 28 |
| 5b | Cache scoping | `acl_scope_hash` in the cache key — otherwise a lower-clearance user hits a higher-clearance answer (ADR-019) | 15 |
| 6 | Budget governor | iterations, depth, wall-clock, tokens, cost | 24, 27 |

Guardrails at point 3 run on **every** entry into the context, not once on the
user's prompt: a tool observation on iteration three is lower-trust than the
authenticated user. Full treatment in [security.md](security.md) §2.

The model may *request*; the application *decides*. A prompt instruction
("do not reveal confidential data") is a hint, never a control.

---

## 6. Human interrupt for risky actions (HITL)

### Risk tiers — classified per action, not per agent

| Tier | Examples | Policy |
| --- | --- | --- |
| `read` | vector/keyword search, read public doc | auto |
| `compute` | calculator, SQL `SELECT`, table extract | auto, sandboxed, timeout, row cap |
| `sensitive-read` | confidential-class document | auto **if** ACL passes, always audited |
| `write` / `external` | send report, write record, call external API | **pause → human approval** |
| `irreversible` | delete, publish, notify executives, financial submission | **pause → approval + second approver** |

### Mechanics — a durable pause, not a blocked thread

```text
agent decides action
   → tool broker classifies risk tier
   → tier ≥ write?
        yes → persist run state (Postgres), A2A task → input-required
              emit approval request {action, args, agent reasoning, trace link,
                                     affected documents, estimated cost}
              agent process releases its slot (no GPU/thread held)
        ← approver: approve / deny / edit-args
              resume by token → run continues from the saved step
   → timeout (default 30 min) = auto-deny, task failed, audited
```

Requirements that follow from "durable":
- run state in Postgres, keyed by `task_id`, survives restart and redeploy
- resume token is single-use and bound to the approver's identity
- the approval record is immutable and part of the attestation (§4 L4)
- approving an action does **not** approve the next one — no blanket consent

### Kill switch and circuit breaker
- cancel a running task (A2A `canceled`) — propagates to all branches
- disable one tool globally (feature flag in Redis) when it misbehaves;
  agents see it as unavailable and must degrade, not retry
- disable an agent version and fall back to the previous one (rollback)

---

## 7. Failure modes introduced by agents (and their detection)

| Failure | Detection | Mitigation |
| --- | --- | --- |
| Infinite loop / oscillation | `llmops_agent_iterations` histogram, repeated identical tool args | max_iterations, loop detector on action hash |
| Cost explosion via fan-out | `llmops_cost_usd_total` per task, budget governor | shared budget debited atomically, branch cap |
| Deadlock in join | task age, `input-required` with no approver | deadline on every branch, auto-deny timeout |
| Prompt injection via a retrieved document | guardrail hit rate, unexpected tool requests | data/instruction separation, tool authz, output scan |
| Prompt injection via an **A2A peer response** | same, at the peer boundary | peer output sanitized and framed as data; never concatenated into the system prompt |
| Fabricated citation | L1 verification failure rate | hard fail before the answer is returned |
| Cascading retries → GPU saturation | queue depth + retry counter | no retry on timeout at agent level; circuit breaker |
| Delegation depth explosion | depth metric | `max_delegation_depth=2` |

---

## 8. What is simulated in this lab

**THIS IS A SIMPLIFICATION** — the A2A peers all run on one machine, in
one trust domain, over plaintext HTTP inside the Compose network, and all
"agents" share one 8 GB GPU through the model gateway. A real deployment
would place each agent in its own security domain with mTLS, per-agent
service identity (SPIFFE or equivalent), independent capacity, and an
external approval workflow (ticketing / chat-ops) instead of a local
approval endpoint. The lab keeps the *interfaces* identical so the
difference is deployment, not design.
