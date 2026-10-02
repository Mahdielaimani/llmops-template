# ADR-019: Authorization model — RBAC for capabilities, ABAC for data, ACL as a query predicate

**Status:** Accepted
**Date:** 2026-10-02
**Phase:** 11 (retrieval ACL), 15 (cache scoping), 16 (edge RBAC), 24 (tool authz)

## Context
`docs/business-requirements.md` §5 requires that a user never retrieves a
document above their clearance (SEC-2, hard gate Q-6 = 0), that external
auditors get a time-boxed scope, and that tool actions are authorized per user
and context (SEC-3). Three different questions hide behind the word
"permissions", and conflating them is the common failure:

- may this user call this *endpoint*?
- may this user see this *document*?
- may this user cause this *action*, right now?

A single role check cannot express "these three documents, for six weeks, for an
external auditor".

Separately, the obvious implementation of document filtering — retrieve, then
drop what the user cannot see — is the weaker one, and the reason is not
obvious enough to leave undocumented.

## Decision

**1. Three layers, named separately.**

| Layer | Question | Model | Where |
| --- | --- | --- | --- |
| RBAC | capability | role → route | Kong (Phase 16) |
| ABAC | data | `classification ≤ clearance ∧ department ∈ user.departments ∧ now() ∈ validity` | retrieval + context assembly |
| Tool authorization | action | attributes + context + risk tier | tool broker (Phase 24) |

**2. The ACL is a predicate inside the retrieval query, not a filter on its
output** — pushed into the Qdrant payload filter and the identical predicate on
the BM25 query — and then **re-verified at context assembly** as an independent
second check.

**3. The response cache key includes the ACL scope**, not just the query text.

**4. Deny is the default.** A document with no ACL metadata, or a tool with no
declared risk tier, is treated as maximally restricted (the tool default is
`write`, per ADR-016).

## Why pre-filter, not post-filter

| Failure of post-filtering | Consequence |
| --- | --- |
| **Leak through absence** — top-20 all restricted, filtered to zero | the user learns matching content exists and is hidden; for M&A or unreleased results that inference is the leak |
| **Silent top-k collapse** — 15 of 20 filtered | the model receives 5 chunks, answer quality drops, nothing errors and no metric fires |
| **Score / count leakage** | any exposed relevance score or result count carries information about unreadable documents |

Pre-filtering removes all three: the disallowed chunk never enters the candidate
set, so `top_k` is honoured within the permitted corpus and there is nothing to
infer from.

## Why the cache needs the ACL scope

```text
User A (clearance: confidential) asks about the M&A memo   → cached
User B (clearance: internal)     asks a near-identical question
                                 → CACHE HIT → served A's answer
```

Perfect enforcement at ⑤, total bypass at the cache. Semantic caching makes it
worse by design, since near-miss queries are supposed to collide. The key is
therefore `hash(normalized_query, acl_scope_hash, model_version, prompt_version,
index_version)`. This is a **Phase 15** requirement, not a Phase 29 one — which
is why it is recorded here, well before the security phase.

## Alternatives considered

| Option | Why not (now) |
| --- | --- |
| RBAC for document access too | cannot express clearance × classification × time-boxing; would need a role per document set |
| **OPA / Rego** | strong: policy as versioned data, decision logs, reviewable by non-engineers. Rejected for now because the policy must become a Qdrant filter, and round-tripping a Rego decision into a vector-store predicate is awkward. Revisit if policy needs non-engineer review |
| **Cedar** | readable, formally verified, good fit; same filter-translation problem, smaller ecosystem today |
| **OpenFGA / SpiceDB (ReBAC)** | the right answer once access depends on *relationships* ("members of the deal team for deal X"). Not yet needed; adds a stateful service and consistency semantics |
| **pgvector + Postgres RLS** | **genuinely stronger** — enforcement below the application, so an app bug cannot leak rows. Rejected only because Qdrant is already chosen (ADR-002) for hybrid retrieval; named explicitly in interviews as the stronger guarantee |
| Post-filter only | the three failures above |
| Let the LLM decide ("only answer from documents the user may see") | not a control; one injected instruction removes it |

## Trade-offs

+ each layer is independently testable, and Q-6 becomes a unit test rather than a hope
+ pre-filtering is also *faster*: fewer candidates to rerank
+ cache scoping closes a bypass that no amount of retrieval correctness would catch
− authorization logic lives in application code, so it is not reviewable by
  non-engineers; OPA is the escape hatch when that becomes a requirement
− the ABAC predicate must be kept in sync across Qdrant filter, BM25 filter and
  the context-assembly re-check — three places, one truth. Mitigated by deriving
  all three from one `AclPredicate` object with its own tests
− `acl_scope_hash` fragments the cache: users with distinct permission sets
  cannot share entries, so hit rate drops. That is the correct trade and the
  cost must be measured in Phase 15 rather than assumed away

## Consequences
- Qdrant payload carries `doc_id, doc_version, chunk_id, classification,
  department, acl_scope, index_version` so the predicate can be expressed
  server-side.
- The BM25 index must carry the same fields; a lexical index without ACL
  metadata is a bypass.
- Phase 15 cache keys include `acl_scope_hash`; a test asserts two clearances
  never share an entry.
- Phase 11 gains the hard-fail regression test for Q-6 (zero unauthorized
  retrievals), and Phase 29 attack simulation #1 tests it adversarially.
- Audit records the ACL scope used for each query, so a leak can be reconstructed
  after the fact.
- Tools and documents without explicit authorization metadata fail closed.
