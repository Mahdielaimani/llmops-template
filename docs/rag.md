# RAG — Ingestion and ACL-Filtered Retrieval

**Status:** LOCAL / PRODUCTION-LIKE. Phase 8. Every number measured by
`benchmarks/rag_retrieval.py` against planted ground truth, not judged.

**Scope boundary:** Phase 8 delivers **retrieval**. `/rag` returns citations and
`answer: null` — generation arrives with the model gateway in Phase 15. Returning
a fabricated answer to look complete would be worse than returning null.

---

## 1. RAG is a pipeline, not "vector DB + LLM"

```text
OFFLINE — ingestion                            ONLINE — retrieval
───────────────────────                        ──────────────────────────
documents (12)                                 question
   │ validate                                     │ embed query (+ bge prefix)
   ▼ parse                                        ▼
   │ clean  (conservative — number                ACL PREDICATE  ◄── ADR-019
   │         formatting is the answer)            │  rendered into the Qdrant
   ▼ extract metadata                             │  filter, not applied after
   │  doc_id · version · classification           ▼
   │  department · effective_date                 dense search (cosine, top-20)
   ▼ chunk  (180 tok target, 40 overlap,          │
   │         2.34 chars/token — financial)        ▼
   ▼ embed  (bge-small-en-v1.5, 384-dim)          ACL RE-CHECK  ◄── independent
   ▼ index  (Qdrant, ACL fields in payload)       │  second enforcement point
       collection = docs_<index_version>          ▼
                                                  context builder (token budget,
                                                  sources framed as DATA)
                                                  │
                                                  ▼
                                                  citations: doc_id@vN#chunk (p.N)
```

Phase 9 inserts BM25 + fusion + reranking between search and the context builder.

---

## 2. The corpus is synthetic, and that is a feature

No real financial documents exist here (`business-requirements.md` §6). Synthetic
data means **facts are planted at known locations**, so retrieval is scored
against ground truth rather than an opinion.

12 documents, 18 chunks, 12 planted facts:

| Property the corpus must have | Why | Instance |
|---|---|---|
| **Versions** | "what did Q2 say *before* the restatement" must have an answer | `QR-EMEA-SALES-Q2FY2026` v1 and v2, different revenue figures |
| **Classifications** | the ACL needs something to enforce | public → confidential, all four levels |
| **Near-duplicates** | retrieval must discriminate, not keyword-match | `BUD-FINANCE-FY2026` and `BUD-EMEA-SALES-FY2026`, same shape, different numbers |
| **A document that must never leak** | Q-6 hard gate, Phase 29 attack #1 | `MA-PROJECT-HELIOS-001`, confidential, corp-dev only |

Each planted fact records `(question, answer, doc_id, doc_version, page,
classification, min_clearance)` — so a correct response is checkable on both
**content** and **citation**, and the ACL is checkable per identity.

```bash
uv run python -m rag_service.corpus   # 12 docs + data/evaluation/planted_facts.json
```

---

## 3. Chunking carries a Phase 4 measurement

```python
target_tokens = 180
overlap_tokens = 40
chars_per_token = 2.34   # measured, docs/inference.md §1
```

The third parameter is the one that matters. Financial text runs **2.34
chars/token against 5.70 for prose**, so a character budget derived from prose
would produce chunks ~2.4× larger in tokens than intended — blowing the context
budget and the prefill cost (`metrics-and-capacity.md` §2) without anything
failing visibly.

Splitting is **paragraph-aware**. Cutting mid-sentence inside a figure
(`EUR 1,234,` | `567.89`) makes the fact unretrievable, so paragraph boundaries
are preferred and only oversized paragraphs are sentence-split with overlap.

Measured: **18 chunks from 12 documents, 297 chars/chunk mean** (≈127 tokens at
the financial ratio — under target because most pages are short).

### The config is a versioned artifact

```text
index_version = sha256(chunk_config_hash | embed_model | corpus_sha)[:12]
collection    = docs_<index_version>
```

`4a0a78edcfa9` for this build. Change the chunking or the embedding model and the
collection name changes, so an index can never be confused about what produced
it (ADR-015, artifact classes 5 and 7). The application resolves the latest
`docs_*`; Phase 10 replaces that with an alias flip.

---

## 4. Embeddings — fastembed, not sentence-transformers

`BAAI/bge-small-en-v1.5`, 384-dim, **67 MB**. ADR-020 has the reasoning; the
short version is that sentence-transformers pulls PyTorch (~1 GB installed) and
disk is this machine's binding constraint at 97%.

Measured:

| | |
|---|---|
| Model load (cold, first run) | 10 868 ms — downloads weights |
| Embed 32 documents | 48.4 ms → **1.51 ms/doc** |
| Embed 1 query | **4.6 ms** |
| Vector norm | 1.0 — L2-normalised, so cosine is a dot product |

One asymmetry worth knowing: **bge models expect an instruction prefix on the
query side only.** `embed_query()` prepends
`"Represent this sentence for searching relevant passages: "`;
`embed_documents()` does not. Applying it to both, or neither, costs recall.
That is why they are separate methods rather than one.

Model load is baked into the Docker image at build time — otherwise the first
request pays 11 s and the readiness probe flaps.

---

## 5. Measured retrieval quality

Against the 12 planted facts, as the identity that can see everything:

| | |
|---|---|
| **Recall@1** | **0.917** |
| **Recall@3** | **1.000** |
| Recall@5 | 1.000 |
| **MRR** | **0.958** |
| Missed | **0** |

Dense retrieval alone already finds every planted fact within the top 3. One
fact is not rank 1, which is what Phase 9's reranker is for — and it is worth
noting now that **there is very little headroom to demonstrate improvement on a
12-fact corpus**. Phase 9 will need harder questions (multi-hop, near-duplicate
disambiguation) or the hybrid-vs-dense comparison will be noise.

### Latency

| Stage | p50 | p95 |
|---|---|---|
| Embed query | 7.12 ms | 8.73 ms |
| Qdrant search | 26.02 ms | 98.48 ms |
| **Total retrieval** | **33.94 ms** | **105.77 ms** |
| Context built | 2 574 chars p50 | |

Against **SLO-5 (retrieval p95 ≤ 600 ms)** there is comfortable headroom — but
the budget is about to be spent: Phase 9's CPU cross-encoder is predicted at
200–400 ms (`metrics-and-capacity.md` §1), which consumes most of what remains.
The reranker, not the vector search, is the retrieval bottleneck. That
prediction is on the record here before it is measured.

---

## 6. ACL is a query predicate — measured

ADR-019 requires the ACL **inside** the query. One `AclPredicate` object renders
three ways: a Qdrant filter, a callable for the re-check, and a cache-scope hash.

```python
allowed = doc.classification <= user.clearance
          and (doc.classification == PUBLIC or doc.department in user.departments)
          and (user.scoped_doc_ids is None or doc.doc_id in user.scoped_doc_ids)
          and user.is_within_validity
```

### The hard gate: Q-6, zero unauthorized retrievals

72 queries — 12 questions × 6 identities:

| Identity | Clearance | Candidates | Returned | **Leaked** |
|---|---|---|---|---|
| contractor | public | 12 | 12 | **0** |
| analyst-emea | internal | 120 | 60 | **0** |
| analyst-apac | internal | 84 | 60 | **0** |
| auditor | restricted | 36 | 36 | **0** |
| auditor-expired | restricted | **0** | **0** | **0** |
| cfo-office | confidential | 216 | 60 | **0** |
| | | | **TOTAL** | **0** |

### The document that must never leak

```text
question: "What is the indicative offer for Project Helios?"

contractor     candidates=1    reaches_memo=False
analyst-emea   candidates=10   reaches_memo=False
auditor        candidates=3    reaches_memo=False
cfo-office     candidates=18   reaches_memo=True     ← authorized
```

Note the **candidate counts differ by clearance** (1 / 10 / 3 / 18). That is the
pre-filter working: the disallowed chunks were never candidates, so `top_k` is
honoured within each user's permitted corpus and there is no post-filter
collapse. A lower-clearance user sees fewer results, never a result they then
had removed.

**Honest note on leak-through-absence.** Pre-filtering removes the *severe* form
(a zero-result response that reveals hidden matching content, `security.md` §2),
but a user can still infer something from result *counts* across many queries.
Mitigating that fully requires padding or rate-limiting the inference channel,
which this lab does not do. Stated rather than claimed away.

### Time-boxed access

`auditor-expired` retrieves **0 candidates** — the predicate renders an
impossible Qdrant condition rather than an absent filter, so a caller who forgets
to check `expired` still gets nothing. This is the case **RBAC cannot express**:
roles have no validity window, which is why document access here is ABAC.

### Cache scoping, ready for Phase 15

All six identities produce distinct scope hashes:

```text
analyst-apac     33ccfec62e30c682
analyst-emea     19ef136a88c511d9
auditor          e07c811cc0937b20
auditor-expired  200f595ce5021c92
cfo-office       1f61214c6ff88b90
contractor       8bdac540718209f5
```

`/rag` returns `acl_scope` on every response. Phase 15 must include it in the
cache key — without it, a lower-clearance user hits a higher-clearance cached
answer and perfect retrieval enforcement is bypassed (ADR-019).

---

## 7. Two independent enforcement points, on purpose

`retriever.search()` re-checks every hit after the filtered query. That is
redundant **by design**: a bug in the Qdrant filter and a bug in the predicate
are unlikely to coincide, and the re-check is the one that catches a payload
written without `classification_level`.

Measured cost of the redundancy: **0.037–0.13 ms** — about 0.3% of retrieval. If
it ever rejects anything, the retriever logs `acl_recheck_rejected` as a bug
rather than silently dropping the chunk, because a disagreement between the two
means one of them is wrong.

Fail-closed is tested: a payload with no `classification` is denied, not treated
as public.

---

## 8. Retrieved content is data, not instructions

`build_context()` wraps each chunk in delimiters and never merges it into the
instruction:

```text
<source id="AUD-2026-014@v1#...p1.c0" title="..." classification="restricted" effective="2026-08-03">
Testing identified EUR 183,420.55 of revenue recognised in the incorrect period...
</source>
```

A retrieved document can contain injected instructions (`security.md` §2, T3) —
the document author is **less** trusted than the authenticated user. The citation
list is returned alongside the context rather than parsed back out of the model's
answer, so Phase 28's output guardrail can check claims against what was actually
supplied.

---

## 9. The API

```bash
docker compose --profile core --profile rag up -d --build

curl -s localhost:8070/whoami -H 'x-user: analyst-emea' | jq
curl -s localhost:8070/index | jq

curl -s localhost:8070/rag -H 'content-type: application/json' -H 'x-user: contractor' \
  -d '{"question":"What is the indicative offer for Project Helios?","top_k":5}' | jq
# → citations: [] — the chunk was never a candidate

curl -s localhost:8070/rag -H 'content-type: application/json' -H 'x-user: cfo-office' \
  -d '{"question":"What is the indicative offer for Project Helios?","top_k":5}' | jq
# → the memo, with its citation
```

| Route | Purpose |
|---|---|
| `POST /rag` | retrieval with citations, `answer: null` until Phase 15 |
| `GET /whoami` | which identity the stub resolved, and its `acl_scope_hash` |
| `GET /index` | collection, index_version, chunk count, embed model |
| `GET /health/live`, `/health/ready` | readiness registers a real Qdrant check |

**Identity is a stub.** `X-User` maps to a fixed table of six users. Phase 16
replaces it with a JWT verified by Kong — the claim *shape* is already what the
token will carry, so the swap is local to `resolve_user()`. Unknown or absent
identity resolves to the least-privileged user: **fails closed**.

---

## 10. Reproduce it

```bash
docker compose --profile core up -d qdrant
uv run python -m rag_service.corpus
uv run python -m rag_service.ingest
uv run python benchmarks/rag_retrieval.py     # exits non-zero if Q-6 is violated
uv run poe test-unit -k rag                    # 18 ACL tests, no Qdrant needed
```

The ACL tests run without a server because `AclPredicate.qdrant_filter()` returns
a plain dict — deliberately, so authorization is unit-testable.

---

## 11. What is not done yet

| Gap | Phase |
|---|---|
| Generation — `answer` is null | 15 (model gateway) |
| BM25, fusion, reranking | 9 |
| Harder eval set; 12 facts leave no headroom to show improvement | 9, 11 |
| Postgres as document system of record; versions live in filenames | 10 |
| Async ingestion via queue; currently a synchronous script | 20 |
| Real JWT instead of the `X-User` stub | 16 |
| Cache with `acl_scope_hash` in the key | 15 |
| Full Recall@k / MRR / context-precision reporting into MLflow | 11, 13 |
| PDF/DOCX parsing — `pypdf` is a dependency but unused; corpus is JSON | 20 |

---

## 12. Interview answer

> "RAG is a pipeline, not a vector database with an LLM attached. Offline:
> validate, parse, clean conservatively — aggressive cleaning destroys the number
> formatting the answers depend on — extract metadata, chunk paragraph-aware so a
> figure never splits mid-number, embed, index. The chunk size is in *tokens* and
> I measured that financial text runs 2.34 characters per token against 5.70 for
> prose, so a character budget borrowed from prose would make chunks 2.4× larger
> than intended and quietly blow the context budget.
>
> The index identity is a hash of the chunking config, the embedding model and a
> corpus snapshot, so an index can never be confused about what produced it.
>
> Online, the part that matters: the ACL is a **predicate inside the retrieval
> query**, not a filter on its results. Post-filtering leaks — if the top twenty
> are all restricted you return zero and the user learns something hidden exists,
> and your top-k silently collapses. I measured 72 queries across six identities
> with four classification levels: zero unauthorized retrievals, and the
> confidential M&A memo is reachable only by the one identity cleared for it.
> Candidate counts differ per clearance, which is the pre-filter working.
>
> Access is attribute-based, not role-based, because an external auditor needs
> specific documents for a fixed window — my expired auditor retrieves zero
> candidates, and a role cannot express that. There's a redundant re-check after
> the query costing 0.1 ms: two independent enforcement points, and if they ever
> disagree that's logged as a bug rather than a silent drop.
>
> Retrieval quality is scored against planted ground truth rather than a judge —
> Recall@3 of 1.0 and MRR 0.958 — and latency is 34 ms p50, of which the vector
> search is 26. The reranker I haven't built yet is predicted at 200–400 ms, so
> it, not the search, will be the retrieval bottleneck."
