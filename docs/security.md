# Security

**Status:** design (Phases 11, 16, 24, 26b, 28, 29). Nothing here is
production-ready. What the lab actually implements versus simulates is stated in
§8 — read that before quoting any of this as "done".

---

## 1. The one principle

**The LLM is never a security boundary.** It may *propose* an answer or an
action; every control that decides whether something is permitted lives outside
the model and cannot be argued with by text.

Wrong:

```text
User → LLM ← "You are a helpful assistant. Never reveal confidential data."
```

That is a hint. One injected instruction in a retrieved document removes it.

Right:

```text
User → authenticate → authorize → filter the data → build context → LLM
```

The same applies to tools: the model emits `call_tool("export_customer_data")`;
the application decides whether that call happens.

Two framings that follow:

- **Zero Trust** — no component trusts another by default, including
  `rag-service` trusting the gateway. Identity is verified at each hop.
- **Defense in depth** — seven enforcement points, each able to fail without
  the system becoming open.

---

## 2. The flow

Audit and runtime observation are **bands, not stages**: a request denied at
authentication must still be audited, and eBPF keeps watching after the
application layer is compromised. The agent path is a **loop**, and every tool
observation re-enters the context as untrusted input.

```text
┌───────────────────── AUDIT — every stage, every DENY ──────────────────────┐
│ ┌────────────── RUNTIME — eBPF, out-of-band, always on ─────────────────┐  │
│ │                                                                       │  │
│ │  User                                                                 │  │
│ │    │  JWT: signature, aud, iss, exp, nbf                              │  │
│ │    ▼                                                                  │  │
│ │  ① AUTHENTICATION  →  identity + attributes                           │  │
│ │    │                   (role, department, clearance, tenant)          │  │
│ │    ▼                                                                  │  │
│ │  ② EDGE: RBAC capability · rate limit · body size · request id        │  │
│ │    ▼                                                                  │  │
│ │  ③ BUDGET GOVERNOR  — iterations · wall clock · tokens · cost ──────┐ │  │
│ │    ▼                                                                │ │  │
│ │  ┌─► ④ INPUT GUARDRAILS — PII, injection heuristics, length         │ │  │
│ │  │      ▼                                                           │ │  │
│ │  │   ⑤ RETRIEVAL — ACL is a PREDICATE IN THE QUERY                   │ │  │
│ │  │      │          (Qdrant payload filter + same filter on BM25)     │ │  │
│ │  │      ▼                                                           │ │  │
│ │  │   ⑤b RE-VERIFY ACL at context assembly  ◄── second, independent   │ │  │
│ │  │      ▼                                                           │ │  │
│ │  │   ⑥ CACHE — key includes identity / ACL scope                     │ │  │
│ │  │      ▼                                                           │ │  │
│ │  │      LLM — proposes: answer | tool_call                           │ │  │
│ │  │      ▼                                                           │ │  │
│ │  │   ⑦ TOOL BROKER — identity → ABAC → allowlist → param schema      │ │  │
│ │  │      │             risk tier ≥ write → durable HITL pause         │ │  │
│ │  │      ▼                                                           │ │  │
│ │  │      TOOL EXEC — sandboxed, timeout, row cap                      │ │  │
│ │  │      ▼                                                           │ │  │
│ │  └──── observation — UNTRUSTED, re-enters ④ ──────────────────────────┘ │  │
│ │         (loop bounded by ③)                                           │  │
│ │      ▼                                                                │  │
│ │   ⑧ OUTPUT SECURITY — schema · PII · secrets ·                        │  │
│ │                        citations validated against the FILTERED set   │  │
│ │      ▼                                                                │  │
│ │    Response                                                           │  │
│ └───────────────────────────────────────────────────────────────────────┘  │
└───────────────────────────────────────────────────────────────────────────┘
```

### Why ACL is a query predicate, not a post-filter

Filtering the retrieval *result* is the common implementation and the weaker
one. Three concrete failures:

| Failure | What happens |
| --- | --- |
| **Leak through absence** | top-20 returns, all restricted, post-filter yields zero. The user has learned that matching content exists and is hidden from them. In M&A or pre-release results, that inference *is* the leak. |
| **Silent quality collapse** | `top_k=20`, 15 filtered, the model receives 5 chunks. Not an error, no alert — just a worse answer, invisibly. |
| **Score leakage** | any exposed relevance score or result count carries information about documents the user cannot read. |

So: the ACL predicate goes **into** the Qdrant filter and the BM25 query, and is
**re-checked** at context assembly. Two independent points, because a bug in a
pre-filter and a bug in a post-check are unlikely to coincide. Recorded in
[ADR-019](adr/ADR-019-authorization-model.md).

### Why the cache needs the ACL in its key

The failure that perfect ACL enforcement does not prevent:

```text
User A  (clearance: confidential)  asks about the M&A memo  → answer cached
User B  (clearance: internal)      asks a near-identical question
                                   → CACHE HIT → served A's answer
```

Semantic caching makes it worse: near-miss queries collide by design. **The
cache key must include the identity or, better, the ACL scope** (a hash of the
permission set), so two users with different clearances cannot share an entry.
This lands in **Phase 15**, long before the security phase, which is why it is
written down now.

### Why the agent loop matters

A linear diagram implies one LLM call and therefore one input-guardrail pass.
An agent calls the model N times, and each iteration ingests new untrusted text:

- retrieved document content
- SQL query results
- external API responses
- an MCP server's tool output
- **another agent's A2A reply**

Injection in an agentic system usually arrives on iteration three, inside data
the agent fetched itself — not in the user's original prompt. Guardrails
therefore run on **every entry into the context**, not once at the front door.
A document author is less trusted than the authenticated user, not more.

---

## 3. The seven enforcement points

| # | Point | Enforces | Fails how | Phase |
| --- | --- | --- | --- | --- |
| ① | Authentication | identity + attributes from a verified token | expired/forged token → 401 | 16 |
| ② | Edge (gateway) | RBAC capability, rate limit, body size | 403 / 429 / 413 | 16, 21 |
| ③ | Budget governor | iterations, wall clock, tokens, cost | run terminated, audited | 24, 27b |
| ④ | Input guardrails | PII, injection heuristics, length — on **every** context entry | block, redact, or score-and-continue | 28 |
| ⑤ | Retrieval filter | ABAC on documents, pre-filter **and** re-verify | disallowed chunk never exists in the candidate set | 11 |
| ⑥ | Cache scoping | identity/ACL in the key | no cross-tenant hit possible | 15 |
| ⑦ | Tool broker | allowlist, ABAC per action, param schema, risk tier → HITL | DENY, audited; model is told only "not permitted" | 24, 28b |
| ⑧ | Output security | schema, PII, secrets, citation validity vs the filtered set | response withheld or regenerated | 28 |

**None of them is the model.** Prompt-injection detection is ④ — a filter that
reduces how often you depend on ⑤ and ⑦, never a replacement for them.

---

## 4. Authorization: three different things

Conflating these is the most common error in this space.

| Layer | Question | Model | Example |
| --- | --- | --- | --- |
| **RBAC** | may this user use this *capability*? | role → endpoint | analysts may call `/rag`; only admins may call `/admin/reindex` |
| **ABAC** (data) | may this user see this *record*? | attributes | `doc.classification ≤ user.clearance AND doc.department ∈ user.departments AND now() ∈ auditor.window` |
| **Tool authorization** | may this user cause this *action*, now? | attributes + context + risk tier | `export_customer_data` allowed for finance-lead, write tier → requires approval |

Document access in finance is **attribute-based**, not role-based: clearance ×
classification × department, plus time-boxed external auditors. Calling that
"RBAC" is a category error — roles cannot express "these three documents, for
six weeks". Where relationships matter ("members of the deal team for deal X"),
it becomes **ReBAC** (Zanzibar-style).

---

## 5. Threat model — AI-platform-specific

Generic web threats (SQLi, XSS, CSRF) still apply and are not listed. These are
the ones this architecture exists to address.

| # | Threat | Vector | Primary control | Backstop |
| --- | --- | --- | --- | --- |
| T1 | **Unauthorized document access** | user queries for content above clearance | ⑤ ACL as query predicate | ⑤b re-verify; ⑧ citation check |
| T2 | **Direct prompt injection** | user instructs the model to ignore policy | ④ heuristics + classifier | ⑦ tool authz; ⑤ filtered context |
| T3 | **Indirect injection via document** | attacker plants instructions in an ingested PDF | ④ on retrieved content; data/instruction separation in the prompt | ⑦; ⑧ |
| T4 | **Indirect injection via tool or peer** | SQL result, API response, MCP output, A2A reply | ④ re-applied per iteration | ⑦; budget ③ |
| T5 | **Cross-tenant cache leak** | semantically similar query from lower clearance | ⑥ ACL in cache key | per-tenant cache namespaces |
| T6 | **Cost exfiltration / self-DoS** | induce unbounded agent loops or huge contexts | ③ budget governor | ② rate limit; queue shedding |
| T7 | **Unauthorized tool execution** | model proposes a destructive call | ⑦ allowlist + ABAC | HITL on write/irreversible |
| T8 | **Data exfiltration in output** | model summarises restricted content into an allowed answer | ⑧ citation validity vs filtered set | ⑤; audit review |
| T9 | **Model/artifact supply chain** | poisoned checkpoint; `pickle`/`joblib` deserialization = RCE | hash-pinned revisions, no `latest`, safetensors over pickle | image scan; runtime eBPF |
| T10 | **Malicious MCP server or tool** | compromised dependency in the tool layer | allowlist + param schema + sandbox | runtime; egress policy |
| T11 | **Internal pivot / SSRF** | reach `rag-service` directly, bypassing the gateway | token validated at every service, mTLS | network policy |
| T12 | **Secret leakage via logs or errors** | provider error body echoes the prompt; a DSN logged | never return upstream bodies; redaction processor | log review |
| T13 | **Prompt/model rollback to a vulnerable version** | revert past a guardrail fix | release manifest pinning all nine artifacts | quality + security gate in CI |
| T14 | **Audit gap on denial** | denied requests not recorded | audit is a band, not a stage | — |

### Trust boundaries

```text
  INTERNET  │  EDGE          │  PLATFORM            │  DATA           │  MODEL
────────────┼────────────────┼──────────────────────┼─────────────────┼──────────
  user      │ gateway:       │ services:            │ Postgres        │ weights
  (hostile) │ authn, RBAC,   │ mutually              │ Qdrant          │ (code —
            │ rate limit     │ authenticated         │ object store    │  T9)
            │                │                      │                 │
  document  │                │ tool broker ◄── the only place an action
  authors   │                │                 becomes real
  (hostile — T3)             │
                             │ MCP / A2A peers (hostile — T4, T10)
```

Everything crossing a boundary left-to-right is untrusted until a control says
otherwise. **The model sits in its own zone: it reads from DATA and writes
proposals, and holds no authority.**

---

## 6. Technology choices and alternatives

The lab's choice is marked **LAB**. Alternatives are listed with what they
actually buy, because "we used X" is a weak interview answer without "and we
rejected Y because".

### Identity provider

| Option | Buys you | Costs |
| --- | --- | --- |
| **Microsoft Entra ID** | enterprise SSO, conditional access, groups already mapped to org structure | Azure-coupled; app registration and consent friction |
| Keycloak | self-hosted, free, full OIDC, fine-grained authz built in | you operate it; HA is real work |
| Auth0 / Okta | fastest to integrate, good DX | per-MAU cost at scale |
| Ory Hydra / Kratos | API-first, composable, lightweight | assemble it yourself |
| AWS Cognito / GCP Identity Platform | cheap inside that cloud | weaker enterprise features |
| **LAB: stub JWT issuer** | zero setup, claims controllable per test | not an IdP — an Entra/Keycloak swap is Phase 29 |

What matters more than the vendor: validating **signature, `aud`, `iss`, `exp`,
`nbf`** — and doing it at **every service**, not only the gateway (T11).

### API gateway / policy enforcement point

| Option | Buys you | Costs |
| --- | --- | --- |
| **Kong OSS (LAB, ADR-007)** | plugins for jwt/acl/rate-limit/correlation-id/otel, DB-less config, KIC on K8s | token-based rate limiting is Enterprise only |
| Envoy + `ext_authz` | most powerful, delegates decisions to an external authz service, xDS | steep; config overkill for a small platform |
| APISIX | Kong-like plugin model | etcd dependency → more RAM |
| Traefik | simplest ingress, great K8s story | weaker authz plugin ecosystem |
| Cloud gateways (AWS API GW, Azure APIM) | managed, integrated WAF | lock-in; harder local parity |
| nginx | fine load balancer | no plugin/authz layer |

### Authorization engine

| Option | Buys you | Costs |
| --- | --- | --- |
| **LAB: in-app ABAC predicates** | translates directly into a Qdrant filter, trivially testable, zero infra | policy lives in code, not reviewable by non-engineers |
| OPA / Rego | policy as versioned data, decision logs, ubiquitous | Rego is a language to learn; a network hop or sidecar |
| Cedar (AWS) | formally verified, readable, fast | newer ecosystem |
| OpenFGA / SpiceDB (Zanzibar) | **ReBAC** — "members of deal team X", deep relationship graphs | another stateful service; consistency semantics to reason about |
| Casbin | embeddable, many models | less expressive for complex ABAC |
| Permit.io / Oso | managed policy + UI | SaaS dependency |

**Decision rule:** attributes only → in-app or Cedar. Relationships and
hierarchies → OpenFGA/SpiceDB. Policy must be auditable by non-engineers → OPA.

### PII detection and redaction

| Option | Buys you | Costs |
| --- | --- | --- |
| **Microsoft Presidio** | open source, many recognizers, custom patterns, anonymizer included | English-centric by default; FP tuning required |
| Azure AI Language PII | managed, multilingual, strong recall | per-call cost; data leaves the cluster |
| AWS Comprehend / Google DLP | same trade; DLP has strong structured-data support | same |
| spaCy NER + regex | full control, cheap, offline | you own the recall problem |
| Nightfall, Skyflow | enterprise DLP, tokenization vaults | cost; integration weight |

**Caveat to state out loud:** no PII detector has recall near 1.0. Presidio is a
reduction in exposure, not a guarantee — which is why ⑤ ACL filtering, not
redaction, is the control for confidential documents.

### Prompt-injection and guardrails

| Option | Buys you | Costs |
| --- | --- | --- |
| **LLM Guard (LAB-candidate)** | input+output scanners in one lib: injection, secrets, toxicity, token limits | latency per scanner; maintenance |
| Lakera Guard | strong managed detection, fast | SaaS, per-call cost, prompts leave |
| Azure AI Content Safety / Prompt Shields | integrated with Azure, jailbreak detection | Azure-coupled |
| AWS Bedrock Guardrails | integrated, policy-based | Bedrock-coupled |
| NVIDIA NeMo Guardrails | programmable dialog rails, Colang | heavier; dialog-flow oriented |
| Rebuff | canary tokens — detects *successful* exfiltration attempts | narrower scope |
| Guardrails AI / Outlines / Instructor | **output** structure guarantees (Outlines constrains generation itself) | schema-focused, not threat-focused |
| **LAB: rules + heuristics first** | zero deps, fully testable, fast | lower recall — stated, not hidden |

**Classifier honesty:** a DistilBERT-style injection classifier produces a risk
score with real false-positive and false-negative rates and is bypassable by
paraphrase. It may *score and route*; it must never be the only thing between a
request and a tool call.

**Structured output** deserves separate mention: `Outlines` or
grammar-constrained decoding makes an invalid schema *impossible* rather than
detected afterwards. That is a stronger control than validation, and it is the
preferred option when the output is machine-consumed.

### Secrets

| Option | Buys you | Costs |
| --- | --- | --- |
| **LAB: env vars + `SecretStr`, `.env` gitignored** | zero infra; redaction in logs | no rotation, no audit, no sealing |
| HashiCorp Vault | dynamic short-lived credentials, rotation, audit log | you operate it; unseal procedure |
| Cloud managers (AWS SM, Azure Key Vault, GCP SM) | managed rotation, IAM-integrated | cloud-coupled |
| External Secrets Operator | syncs cloud secrets into K8s | one more controller |
| SOPS + age / Sealed Secrets | secrets in Git, encrypted — good GitOps fit | no rotation; key management on you |

### Supply chain and container security

| Concern | LAB | Alternatives |
| --- | --- | --- |
| Dependency CVEs | `pip-audit` / `uv` lock + CI | Snyk, Dependabot, Renovate |
| Image CVEs | Trivy in CI | Grype, Clair, Snyk Container |
| SBOM | Syft | CycloneDX tooling |
| Image signing | Cosign / Sigstore | Notary v2 |
| Admission control | — | Kyverno, OPA Gatekeeper |
| Non-root, pinned base | already done in both Dockerfiles | distroless, Chainguard images |
| **Model artifacts** | **safetensors, pinned revision + hash** | — never `pickle`/`joblib` from an untrusted source (T9) |

### Runtime security

| Option | Buys you | Costs |
| --- | --- | --- |
| **Tetragon (eBPF)** | kernel-level process/network/file observability **and enforcement**, low overhead | eBPF-capable kernel; policy authoring |
| Falco | mature, large rule corpus, CNCF | detection-focused; more overhead historically |
| Sysdig Secure / Aqua / Prisma | managed, compliance reporting | cost |
| seccomp + AppArmor + read-only rootfs + dropped caps | free, huge return, works everywhere | static; no behavioural detection |

**Order of operations:** seccomp/AppArmor/non-root/read-only first — they cost
nothing and remove whole classes. eBPF is the layer that still helps *after*
something is compromised, which is exactly why it is listed last and not
skipped.

### Audit log

| Option | Buys you | Costs |
| --- | --- | --- |
| **LAB: append-only Postgres table** | transactional with the action, easy to query | not tamper-proof against a DB admin |
| ClickHouse / OpenSearch | scale, retention, fast analytics | another stateful system |
| Object storage with WORM / Object Lock | genuine immutability | query ergonomics |
| Cloud-native (CloudTrail-style) | managed, integrated | only covers cloud control plane, not your app semantics |

Required shape, every entry: **who · what · when · which resource or tool ·
result · release_id · request_id**. Including the **denials**, which are the
entries that matter most.

### Human-in-the-loop / durable approval

| Option | Buys you | Costs |
| --- | --- | --- |
| **LAB: run state in Postgres + resume token** (ADR-016) | survives restart, releases the worker slot, no new infra | you write the state machine |
| Temporal | durable execution as a product, retries, visibility | a cluster to run; new programming model |
| LangGraph `interrupt()` + checkpointer | native to the agent framework | couples HITL to that framework |
| Step Functions / Durable Functions | managed orchestration | cloud-coupled |
| Slack / Teams / ServiceNow approval | where approvers already are | integration + identity mapping |

**Non-negotiable regardless of choice:** the pause is **durable**, the worker
slot is **released**, the resume token is **single-use and bound to the
approver**, and a timeout **auto-denies**.

### Vector-store tenancy and ACL

| Option | Buys you | Costs |
| --- | --- | --- |
| **Qdrant payload filter (LAB)** | ACL as a query predicate, exactly the pattern §2 requires | filter correctness is on you |
| pgvector + Postgres **RLS** | authorization enforced by the database itself — strongest option | pgvector performance ceiling at scale |
| Weaviate multi-tenancy | tenant isolation as a first-class concept | tenant ≠ document-level ACL |
| Pinecone namespaces / Milvus partitions | hard separation per namespace | coarse; poor fit for per-document clearance |

**Postgres RLS is worth naming in an interview**: it moves enforcement below the
application, so an application bug cannot leak rows. That is a stronger
guarantee than any filter you write yourself.

---

## 7. Four attack simulations (Phase 29 acceptance)

Each must be *detected*, *blocked*, and *regression-tested* — a one-off fix
without a test is not a mitigation.

1. **Unauthorized document access.** Analyst queries for a confidential M&A
   memo. Expected: the chunk never enters the candidate set (⑤); audit records a
   denial; zero leakage via result count or latency. Hard gate: Q-6 = 0.
2. **Prompt injection, direct and indirect.** A payload in the user message and
   a second planted inside an ingested PDF, each instructing exfiltration.
   Expected: ④ flags, ⑦ refuses the tool call, ⑧ blocks the output; the injected
   instruction never acquires authority.
3. **Malicious tool request.** Model proposes `delete_document` /
   `export_customer_data`. Expected: ⑦ DENY on allowlist and ABAC; for write
   tier, durable HITL pause that survives `docker restart`.
4. **Excessive requests / unbounded loop.** Burst traffic plus an adversarial
   goal designed to loop. Expected: ② rate-limits with 429, ③ terminates on
   iteration/cost budget, queue sheds rather than exhausting VRAM.

Plus, promoted from §5 because it is cheap to test and catastrophic to miss:

5. **Cross-tenant cache leak.** Two users, different clearance, semantically
   near-identical queries. Expected: no shared cache entry. Belongs in **Phase
   15**, not 29.

---

## 8. What the lab implements, and what it only models

**THIS IS A SIMPLIFICATION.** Being specific about the gap is the point.

| Control | Lab | Production would add |
| --- | --- | --- |
| Authentication | stub JWT issuer, real validation | Entra ID / Keycloak, conditional access, MFA |
| Service-to-service | plaintext HTTP on the Compose network | mTLS, SPIFFE workload identity, network policy |
| ABAC | predicates in Python, unit-tested | OPA/Cedar with decision logs, or Postgres RLS |
| PII | Presidio or regex, tuned on synthetic data | tuned per locale, human review loop |
| Injection defense | heuristics, honest recall | layered managed detection + canaries |
| Secrets | env + `SecretStr` | Vault with short-lived dynamic credentials |
| Audit | append-only Postgres | WORM storage, separate retention domain, SIEM |
| Runtime | seccomp/non-root; eBPF documented | Tetragon or Falco enforcing, centrally managed |
| Tenancy | one tenant, ACL fields present | per-tenant keys, separate indexes, DR per tenant |
| Corpus | synthetic documents with planted facts | real data under real classification, and a DPIA |

The interfaces are kept identical so the difference is deployment, not design.

---

## 9. Explaining this in an interview

A complete spoken version lives in
[docs/interview/security-layer.md](interview/security-layer.md). The compressed
version:

> "I treated security as a cross-cutting layer from identity through to the
> model's output, with one rule: the LLM is never a security boundary. It can
> propose an answer or a tool call, but every control that decides what is
> permitted sits outside it and can't be talked out of its decision.
>
> Identity comes from an OIDC provider and the token is validated at every
> service, not just the gateway — otherwise an internal pivot reaches an
> unauthenticated service. From the claims I get role, department and clearance.
> The gateway does coarse RBAC, rate limiting and request size.
>
> Document access is attribute-based, not role-based — clearance against
> classification, department, and a validity window for external auditors — and
> the critical detail is that the ACL is a **predicate inside the retrieval
> query**, not a filter on its results. Post-filtering leaks: if the top twenty
> are all restricted you return zero results and the user has just learned that
> something exists and is hidden, and your top-k silently collapses with no
> alert. So it's pushed into the Qdrant filter and the BM25 query, then
> re-verified at context assembly.
>
> Retrieved content is **untrusted data, not instructions** — and so is every
> tool result, MCP response and peer-agent reply. In an agent loop the model is
> called repeatedly, so guardrails run on every entry into the context, not once
> at the front door. Real injection in agentic systems arrives on iteration
> three, inside data the agent fetched itself.
>
> For tools, the model emits a proposed call and the application intercepts it:
> identity, then ABAC on that specific action in that context, then a parameter
> schema. Actions are classified by risk tier, and write or irreversible tiers
> suspend the run durably — state in Postgres, worker slot released — until a
> human approves with a single-use token. Timeout auto-denies.
>
> Two things I'd add that people usually miss. First, the cache key has to
> include the ACL scope: two users with different clearances asking semantically
> similar questions will otherwise share a cache entry, and perfect ACL
> enforcement is bypassed. Second, an unbounded agent loop is a vulnerability,
> not just a bug — it's cost exfiltration and a denial of service against shared
> GPU capacity — so there's a budget governor on iterations, wall clock, tokens
> and cost.
>
> Output is validated for schema, PII and secrets, and citations are checked
> against the ACL-filtered set, because the subtle leak is the model summarising
> restricted content into an answer that looks permitted.
>
> Underneath, eBPF via Tetragon observes process and network behaviour, which is
> the layer that still helps once something is compromised — after the cheap
> wins of non-root, read-only rootfs, dropped capabilities and seccomp.
> Everything, including every denial, lands in an append-only audit log: who,
> what, when, which resource or tool, result, and the release id.
>
> So: prompt-injection detection is a defence layer, RBAC covers capabilities,
> ABAC covers data, tool authorization covers actions — and the model can
> propose, but never authorize itself."

---

## 10. Related

- [agent-platform.md](agent-platform.md) §5, §6 — enforcement points, HITL risk tiers
- [ADR-016](adr/ADR-016-hitl-risk-tiers.md) — durable human interrupt
- [ADR-019](adr/ADR-019-authorization-model.md) — RBAC/ABAC split, pre-filter, cache scoping
- [business-requirements.md](business-requirements.md) §5 — SEC-1…9, Q-6, Q-7
- [versioning.md](versioning.md) §8 — rollback past a security fix (T13)
