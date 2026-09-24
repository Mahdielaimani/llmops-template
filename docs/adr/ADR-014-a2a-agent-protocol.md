# ADR-014: A2A for agent-to-agent communication, alongside MCP

**Status:** Accepted
**Date:** 2026-09-24
**Phase:** 26b

## Context
Phase 27 introduces a supervisor delegating to research, financial,
compliance and writer agents. Those peers are not functions: each has its
own model, prompt, context window, tool set and budget, and a delegated
task can run for minutes, need a human approval mid-flight, or be
cancelled. MCP (Phase 26) models a *tool call* — short, request/response,
stateless from the caller's view — and does not express task lifecycle,
peer discovery, or resumability.

Without a protocol the alternative is ad-hoc HTTP between agents: every
pair invents its own status encoding, there is no discovery, and a
supervisor restart loses in-flight delegations.

## Decision
Adopt **A2A** for agent↔agent, keeping MCP for agent↔tool.

- Each agent exposes an A2A endpoint plus an **Agent Card** at
  `/.well-known/agent-card.json`, generated from the skill registry
  (ADR: skills are the source of truth, the card is a build artifact).
- Work is a **task** with lifecycle `submitted → working →
  input-required → completed | failed | canceled`.
- Task state, including the accumulated context and the resume point, is
  persisted in **Postgres** keyed by `task_id`. Agents hold no in-memory
  run state, so a restart or a pending approval is survivable and the
  worker slot is released while waiting.
- Streaming uses HTTP+SSE, reusing the token-streaming path already built
  for `/chat`.
- Peer authorization (which agent may call which, with which scopes) and
  sanitization of peer responses are enforced by our application.

## Alternatives considered
| Option | Why not |
|---|---|
| Plain REST between agents | reinvents lifecycle, discovery, cancellation, streaming; no interop |
| MCP for agent↔agent too | models a stateless tool call; no task state, no long-running/resumable semantics, no peer discovery |
| A message bus only (Redis streams / Kafka) | good transport, but no request/response semantics, no discovery, no standard task states; we still use Redis for the blackboard |
| Framework-native handoffs (LangGraph/CrewAI/AutoGen in-process) | couples all agents into one process and one deploy unit; kills independent scaling, independent rollback, and the "separate trust domain" lesson |
| gRPC | fine transport; still no agent semantics, and adds tooling weight for a lab |

## Trade-offs
+ agents become independently deployable, versioned and rollable-back units
+ task lifecycle gives cancellation, durable pause (HITL) and observability for free
+ Agent Card makes routing by *capability* possible instead of by hostname
− a second protocol to learn and instrument next to MCP
− network hop and serialization per delegation (measured; small next to decode)
− the protocol tempts people to treat peers as trusted — explicitly rejected below

## Consequences
- **A2A is not a security boundary.** A peer response is untrusted input:
  sanitized, framed as data, never concatenated into a system prompt.
  Peer authz is a table in our app, checked per call.
- Postgres becomes part of the agent hot path → a readiness check and a
  failure mode (Incident 4) now affect agent runs, not just RAG.
- Every delegation is a span; `task_id` joins traces across agents.
- HITL (ADR-016) reuses `input-required` rather than inventing a state.
- **THIS IS A SIMPLIFICATION:** locally all peers share one trust domain,
  one GPU and plaintext HTTP inside the Compose network. Production would
  add mTLS, per-agent workload identity, and independent capacity.
