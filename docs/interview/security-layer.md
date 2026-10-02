# Interview answer — "How did you apply the security layer on your AI platform?"

Three versions: **90 seconds** (the opener), **5 minutes** (the full walk), and
**follow-ups** with the technology alternatives an interviewer will probe for.

Full design: [../security.md](../security.md). Decisions:
[ADR-016](../adr/ADR-016-hitl-risk-tiers.md),
[ADR-019](../adr/ADR-019-authorization-model.md).

---

## Version 1 — the 90-second opener

> I treated security as a cross-cutting layer, from the user's identity through
> to the model's output, on one rule: **the LLM is never a security boundary.**
> It can propose an answer or a tool call, but every control that decides what is
> permitted sits outside the model and can't be talked out of its decision by
> text.
>
> Concretely there are seven enforcement points. Authentication gives me
> identity and attributes. The gateway does coarse RBAC, rate limiting and
> request size. A budget governor caps iterations, wall clock, tokens and cost.
> Input guardrails handle PII and injection — on every entry into the context,
> not just the user's prompt. Retrieval applies document ACLs **as a predicate
> inside the query**, so a disallowed chunk never exists in the candidate set.
> The cache is keyed by ACL scope. A tool broker authorizes each proposed action,
> with a durable human approval for anything that writes. Output is validated for
> schema, PII and citation validity against that filtered set.
>
> Audit and runtime observation aren't stages in that chain — they're bands
> across all of it, because a request denied at authentication is the most
> security-relevant event I have and it still has to be recorded.
>
> So: prompt-injection detection is a defence layer. RBAC covers capabilities,
> ABAC covers data, tool authorization covers actions. The model proposes; it
> never authorizes itself.

---

## Version 2 — the 5-minute walk

### Opening frame

> Two principles. **Zero Trust** — no component trusts another by default,
> including my own services trusting my own gateway. And **defense in depth** —
> several independent controls, each able to fail without the system becoming
> open. The specific rule for an AI platform is that the model is not one of
> those controls.

### 1. Authentication and identity

> Identity comes from an OIDC provider — I used **Microsoft Entra ID** because
> the organisation's groups already map to departments and clearance, and
> conditional access comes for free. From the token I get role, department,
> clearance and tenant.
>
> The detail that matters more than the vendor: the token is validated at
> **every service**, not only at the gateway — signature, audience, issuer,
> expiry, not-before. If only the edge validates, then any internal pivot or
> SSRF reaches an unauthenticated service, and that's not Zero Trust, that's a
> hard shell around a soft centre.

### 2. Edge

> The gateway — **Kong** — handles coarse RBAC, rate limiting, body size limits
> and request-id propagation. Coarse means "may this role call `/rag` at all",
> not "may this user see this document". Those are different questions and
> conflating them is the usual mistake.

### 3. Three kinds of authorization

> **RBAC** answers *may this user use this capability* — role to endpoint.
>
> **ABAC** answers *may this user see this record* — and document access in
> finance is attribute-based, not role-based: `classification ≤ clearance AND
> department ∈ user.departments AND now() ∈ validity_window`. That last clause is
> why roles don't work: an external auditor needs three specific documents for
> six weeks, and you cannot express that as a role without inventing one per
> engagement.
>
> **Tool authorization** answers *may this user cause this action, right now* —
> attributes plus context plus a risk tier.

### 4. Retrieval — the part most people get wrong

> The ACL is a **predicate inside the retrieval query**, not a filter on its
> results. Pushed into the Qdrant payload filter and the identical predicate on
> the BM25 side, then re-verified at context assembly as an independent second
> check.
>
> Post-filtering is the common implementation and it leaks three ways.
> **Leak through absence** — if the top twenty are all restricted you return zero
> results, and the user has just learned that matching content exists and is
> hidden from them; for M&A or unreleased results that inference *is* the leak.
> **Silent top-k collapse** — ask for twenty, filter fifteen, the model gets five
> chunks, the answer degrades and nothing errors, so no metric fires.
> And **score or count leakage** if either is ever exposed.
>
> Pre-filtering removes all three, and it's faster — fewer candidates to rerank.

### 5. Retrieved content is data, not instructions

> Anything the system fetches is untrusted: document content, SQL results,
> external API responses, MCP tool output, and another agent's reply over A2A.
> It goes into the prompt framed as data, never concatenated into the system
> instruction.
>
> And because an agent is a **loop**, not a single call, guardrails run on every
> entry into the context. Real injection in agentic systems arrives on iteration
> three, inside data the agent fetched itself — not in the user's original
> message. A document author is *less* trusted than the authenticated user, not
> more.

### 6. Tools — propose versus execute

> The model emits something like `call_tool("export_customer_data")`. It cannot
> execute it. The application intercepts and checks identity, then ABAC on that
> specific action in that context, then a parameter schema, then resource limits
> — timeout, row cap, sandbox.
>
> Actions carry a **risk tier**: read and compute go through automatically;
> write and irreversible suspend the run for human approval. The suspend is
> **durable** — run state in Postgres, worker slot released — so a pending
> approval survives a restart or a deploy and doesn't hold capacity. The resume
> token is single-use and bound to the approver, and a timeout auto-denies.
> Approving one action never implies consent for the next.

### 7. Two things people usually miss

> **The cache key has to include the ACL scope.** Otherwise: user A with
> confidential clearance asks about the M&A memo, it's cached; user B with
> internal clearance asks a semantically similar question, cache hit, served A's
> answer. Perfect ACL enforcement, complete bypass. Semantic caching makes it
> worse because near-miss collision is the feature.
>
> **An unbounded agent loop is a vulnerability, not just a bug.** It's cost
> exfiltration, and on shared GPU capacity it's a denial of service against other
> tenants. So there's a budget governor on iterations, delegation depth, wall
> clock, tokens and cost, debited atomically across parallel branches.

### 8. Output

> Schema validation, PII and secret scanning, and citation validity checked
> against the ACL-filtered set — because the subtle leak isn't quoting a
> forbidden document, it's the model summarising restricted content into an
> answer that looks permitted. Where output is machine-consumed I'd prefer
> constrained decoding over validation, so an invalid schema is impossible rather
> than detected afterwards.

### 9. Runtime and supply chain

> Cheap wins first: non-root, read-only root filesystem, dropped capabilities,
> seccomp, pinned base images. Then **Tetragon on eBPF** for process, network and
> file observability — that's the layer that still helps once an application
> component is already compromised, which is the whole reason to have it.
>
> Supply chain is AI-specific here: model weights are hash-pinned at a specific
> revision, never `latest`, and **safetensors rather than pickle** — a `pickle`
> or `joblib` artifact is executable code, so loading a checkpoint from an
> untrusted source is remote code execution. Same reasoning for MCP servers: a
> compromised tool server is a tool-layer compromise, and MCP is a connectivity
> protocol, not a trust boundary.

### 10. Audit

> Append-only, every entry: **who, what, when, which resource or tool, result**,
> plus the release id and request id so it ties back to exactly which prompt,
> model and index version were serving. Crucially it covers **denials** — a
> request rejected at authentication is the most security-relevant event there
> is, and if audit is the last stage in a pipeline it never gets recorded. So
> audit is a band across every stage, not a step at the end.

### Closing

> The separation is the whole design: the model reasons, and independent
> mechanisms decide. Injection detection reduces how often I have to rely on the
> downstream controls; it is not what makes the system safe. What makes it safe
> is that the model's output is never accepted as an authorization decision.

---

## Follow-ups, with alternatives

An interviewer who knows this space will push on tool choices. The pattern that
scores: **what I chose · what it buys · what I rejected and why · what I'd
change at a different scale.**

### "Why Entra ID?"

> Organisation's groups already encoded department and clearance, and
> conditional access and MFA came with it. **Keycloak** if self-hosting is
> required — full OIDC, free, but you own HA and upgrades. **Auth0 or Okta** for
> fastest integration, at per-MAU cost. **Ory** if I wanted API-first and
> composable. Honestly the vendor matters less than validating the token at
> every service.

### "Why Kong and not Envoy?"

> Kong gave me jwt, acl, rate-limiting, correlation-id and OTel as plugins with
> declarative DB-less config, and Kong Ingress Controller carries the same
> objects to Kubernetes. **Envoy with `ext_authz`** is more powerful and is what
> I'd use if authorization decisions needed to be delegated to an external policy
> service — but xDS is a lot of machinery for a platform this size. **APISIX** is
> comparable with an etcd dependency; **Traefik** is simpler but weaker on
> authorization plugins. One honest limitation: Kong's *token*-based rate
> limiting is Enterprise, so token budgets live in a thin application-side policy
> layer.

### "Where does policy live? Why not OPA?"

> Today the ABAC predicate is application code, because it has to become a
> Qdrant filter and round-tripping a Rego decision into a vector-store predicate
> is awkward. **OPA** is the right move the moment policy needs to be reviewed by
> non-engineers or audited as versioned data with decision logs. **Cedar** if I
> wanted readability and formal verification. **OpenFGA or SpiceDB** — Zanzibar
> style — is the correct answer as soon as access depends on *relationships*
> rather than attributes: "members of the deal team for deal X" is a graph query,
> not a predicate.

### "Is there a stronger way to enforce document ACLs?"

> Yes, and I'd name it: **pgvector with Postgres row-level security**.
> Enforcement moves below the application, so an application bug cannot leak
> rows. I used Qdrant because I wanted hybrid dense-plus-lexical retrieval and
> payload filtering at scale, and RLS with pgvector has a performance ceiling —
> but RLS is the stronger guarantee and that trade should be stated, not hidden.
> **Weaviate** multi-tenancy or **Pinecone** namespaces give hard tenant
> separation but are too coarse for per-document clearance.

### "How do you detect prompt injection, and how good is it?"

> Rules and patterns for known attacks, and a small classifier producing a risk
> score, routed by policy — block, escalate, or continue. And I'd state the
> limitation plainly: a DistilBERT-class injection classifier has real false
> positive and false negative rates and is bypassable by paraphrase. It may
> **score and route**; it must never be the only thing between a request and a
> tool call.
>
> Alternatives: **LLM Guard** for an open-source input/output scanner suite,
> **Lakera Guard** for stronger managed detection at a per-call cost with prompts
> leaving the boundary, **Azure Prompt Shields** or **Bedrock Guardrails** if
> already in that cloud, **NeMo Guardrails** when I need programmable dialog
> rails, and **Rebuff**'s canary tokens to detect *successful* exfiltration
> rather than attempts. They layer; none is a boundary.

### "PII detection — what and how reliable?"

> **Microsoft Presidio** — open source, extensible recognizers, anonymizer
> included, and it runs in-cluster so sensitive text doesn't leave. **Azure AI
> Language PII**, **AWS Comprehend** or **Google DLP** for managed multilingual
> recall at a per-call cost and a data-residency conversation. **spaCy plus
> regex** if I need full control and offline operation.
>
> The caveat I'd volunteer: no detector has recall near 1.0. Presidio reduces
> exposure; it is not the control for confidential documents. ACL filtering is.

### "Secrets?"

> Environment variables with typed secret wrappers and redaction in the log
> pipeline is the floor, and it's what the lab does — but it has no rotation and
> no audit. **Vault** for dynamic short-lived credentials and an audit log,
> **cloud secret managers** when already in that cloud, **External Secrets
> Operator** to sync them into Kubernetes, **SOPS or Sealed Secrets** for a
> GitOps fit at the cost of manual key management. And the log-side rule: never
> return an upstream provider's error body to a caller, because it can echo the
> prompt back.

### "Runtime — why eBPF?"

> Order of operations matters. Non-root, read-only root filesystem, dropped
> capabilities and seccomp cost nothing and remove whole classes of attack — do
> those first. **Tetragon on eBPF** then gives kernel-level process, network and
> file visibility *and* enforcement at low overhead, which is what still helps
> after an application component is compromised. **Falco** is the mature
> alternative with a large rule corpus; **Sysdig**, **Aqua** or **Prisma** if I
> want managed plus compliance reporting.

### "How is the audit log tamper-resistant?"

> In the lab it's an append-only Postgres table, transactional with the action —
> easy to query, but not tamper-proof against a database admin. Production wants
> **WORM object storage** or object lock for genuine immutability, with
> **ClickHouse or OpenSearch** for query ergonomics and a separate retention
> domain so the people who can act are not the people who can delete the record
> of acting.

### "Human-in-the-loop — how, without blocking workers?"

> Durable suspend: run state persisted, worker slot released, resume by
> single-use token bound to the approver, timeout auto-denies. In the lab that's
> Postgres plus an A2A task in `input-required` state. **Temporal** is the
> product answer for durable execution with retries and visibility, at the cost
> of a cluster and a new programming model. **LangGraph's `interrupt()`** with a
> checkpointer is native if already in that framework, at the cost of coupling
> HITL to it. In production the approval surfaces where approvers already are —
> **Slack, Teams or ServiceNow** — rather than a bespoke endpoint.

### "What would you do differently with more time or budget?"

> In order: mTLS with workload identity between services so Zero Trust is real
> at the service boundary; Postgres RLS or OPA so authorization is enforced or
> reviewed outside application code; WORM audit storage; constrained decoding
> instead of output validation wherever output is machine-consumed; and a real
> red-team exercise against the agent loop, because that's where the
> architecture is least proven.

---

## Two traps to avoid saying

1. **Don't lead with the injection classifier.** It invites "so what happens
   when it's bypassed?" Lead with "the model holds no authority", and the
   classifier becomes one layer among several rather than the thing holding the
   system up.
2. **Don't say "we sanitize the prompt so the model can't leak data".** Say
   "the model never receives data the user isn't entitled to". The first is a
   mitigation; the second is an architecture.
