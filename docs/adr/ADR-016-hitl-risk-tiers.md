# ADR-016: Risk tiers and a durable human-in-the-loop interrupt

**Status:** Accepted
**Date:** 2026-09-24
**Phase:** 28b

## Context
Agents (Phase 24+) choose actions at runtime. Reading a document is
recoverable; sending a report to the CFO office, writing a record, or
calling an external API is not. Guardrails (Phase 28) can *block* known-bad
actions but cannot decide whether a legitimate, well-formed, irreversible
action should happen right now — that is a human judgement.

Naive HITL blocks a worker thread (or an inference slot) while waiting for
a human, so a handful of pending approvals exhausts capacity, and a deploy
or crash loses every pending run.

## Decision
1. **Classify actions, not agents**, into five risk tiers: `read`,
   `compute`, `sensitive-read`, `write`/`external`, `irreversible`.
   The tier is declared on the tool/skill and enforced by the tool broker.
   `read`/`compute`/`sensitive-read` proceed automatically (the last one
   only after the ACL check, always audited). `write` requires one
   approval; `irreversible` requires approval plus a second approver.
2. **Durable suspend.** On a gated action the run state is persisted to
   Postgres, the A2A task moves to `input-required`, and the worker
   releases its slot. Nothing is held in memory, so a restart, redeploy or
   30-minute wait costs nothing.
3. **Resume by single-use token**, bound to the approver identity; the
   approver sees the action, its arguments, the agent's stated reasoning,
   the affected documents, the estimated cost, and a link to the trace.
   Options: approve / deny / edit-args.
4. **Timeout = auto-deny** (default 30 min), task failed, audited.
   Approving one action never implies consent for the next.
5. **Kill switch:** cancel a task (propagates to all fan-out branches) and
   a per-tool circuit breaker in Redis that makes a misbehaving tool
   globally unavailable; agents must degrade, not retry.

## Alternatives considered
| Option | Why not |
| --- | --- |
| Prompt the model to "ask before risky actions" | the model is not a control; one injected instruction removes it |
| Approve per agent / per session | blanket consent; the risky action is a property of the action, not the actor |
| Block the coroutine and await approval in memory | holds a slot per pending approval; lost on restart; does not scale past a few |
| Confirm every tool call | approval fatigue → rubber-stamping, which is worse than no gate |
| Post-hoc review only | fine for audit, useless for irreversible actions |

## Trade-offs
+ irreversible actions get a real control point, independent of the model
+ durable state means pending approvals survive deploys — a prerequisite,
  not a nicety, for approvals measured in minutes
+ the same mechanism serves audit (who approved what, with what context)
− added latency and operational load on write-tier actions
− Postgres is now on the agent hot path (Incident 4 gets broader blast radius)
− tier classification is a judgement call that must be reviewed as tools are added

## Consequences
- Every tool declares `risk_tier` in its skill definition; an undeclared
  tool defaults to `write` (fail-closed).
- `input-required` from ADR-014 is reused; no new task state invented.
- The approval record is immutable and part of the answer's attestation
  (`docs/agent-platform.md` §4 L4).
- Phase 28b acceptance test: agent hits a write action → suspends →
  `docker restart agent-service` → approval granted → run resumes and
  completes with a correct result.
- **THIS IS A SIMPLIFICATION:** approvals arrive on a local HTTP endpoint.
  Production would route them to a ticketing or chat-ops workflow with the
  organisation's own identity and separation-of-duties rules.
