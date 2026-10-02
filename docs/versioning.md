# Versioning — Everything That Can Change the Answer

**Status:** design (Phases 10, 14, 35). Enforced progressively.

Rule: **if it can change the answer, it is versioned, stamped on every
response and log line, and rollable back independently.** "The code is in
Git" covers maybe 20 % of what moves in an LLM system.

---

## 1. The nine versioned artifact classes

| # | Artifact | Version key | System of record | Rollback | Breaking? |
| --- | --- | --- | --- | --- | --- |
| 1 | Code | git sha | Git | revert + redeploy image | no |
| 2 | Prompt | `name@semver` + content sha256 | `prompts/**` in Git, registry in MLflow | move the `production` pointer | no |
| 3 | Model | `name:quant@revision` + weights sha | `models/registry.yaml` + MLflow | pointer swap in model-gateway | no |
| 4 | Embedding model | `name@revision` + dim | registry | **requires reindex** | **YES** |
| 5 | Chunking / retrieval config | config sha256 (size, overlap, splitter, top_k, weights, reranker) | Git | reindex only if chunking changed | partial |
| 6 | Corpus document | `doc_id@version` + content sha | Postgres + object store | query previous version | no |
| 7 | Index | `index_version = sha(embed_model, chunk_cfg, corpus_snapshot)` | Qdrant collection + alias | **alias flip** | — |
| 8 | Eval dataset | `dataset@version` | Git + MLflow | pin per run | no |
| 9 | Agent / skill definition | `agent@semver`, `skill@semver` (prompt, tools, limits, risk tier) | Git registry | pointer swap | no |

Plus **config** (runtime settings) — versioned as part of the release
manifest, not independently, because a setting change with no artifact
change must still be reproducible.

---

## 2. The release manifest — the one thing that means "production"

A single immutable object pinning all nine. It is what is deployed, what
is logged, and what "reproduce yesterday's production" resolves to.

```yaml
# releases/2026-09-24T10-15-00Z.yaml
release_id: rel_20260924_1015
created_at: 2026-09-24T10:15:00Z
created_by: mahdi
code:            {git_sha: 784757a..., image: llmops/llm-application:0.4.2}
prompts:
  rag_answer:    {version: 3.2.0, sha256: 9af1...}
  judge_faithful:{version: 1.4.0, sha256: 2b70...}
models:
  primary:       {name: Qwen2.5-7B-Instruct-AWQ, quant: int4, revision: a1b2c3, sha256: ff01...}
  fallback:      {name: gpt-4.1-mini, provider: openai}
embedding:       {name: bge-m3, revision: d4e5f6, dim: 1024}
retrieval:       {chunk_size: 400, overlap: 60, top_k: 20, rerank_k: 5,
                  dense_weight: 0.6, bm25_weight: 0.4, reranker: bge-reranker-v2-m3,
                  config_sha: 7c1d...}
index:           {collection: docs_v7, index_version: 7c1d9a..., corpus_snapshot: 2026-09-23}
agents:
  supervisor:    {version: 1.2.0}
  financial:     {version: 1.3.0, skills: [variance_analysis@2.1.0, financial_calc@1.0.3]}
eval:            {dataset: core_qa@1.6.0, results_run_id: mlflow:/abc123}
config_sha:      e91b...
approved_by:     [mahdi]
quality_gate:    {recall_at_5: 0.83, faithfulness: 0.88, citation_validity: 0.93, p95_s: 10.8}
```

Every response and every log line carries `release_id` (plus the
individual `model_version`, `prompt_version`, `index_version` for
slicing). Phase 22 dashboards break every metric down by these labels —
that is how "quality dropped" becomes "quality dropped for
`rag_answer@3.2.0` only".

---

## 3. Independent rollback

Each artifact rolls back alone; that is the point of versioning them
separately. Ranked by cost:

| Change | Rollback | Time | Risk |
| --- | --- | --- | --- |
| Prompt | move `production` pointer to previous version | seconds | low |
| Agent / skill | pointer swap | seconds | low |
| Model | model-gateway routes to previous model (kept warm if VRAM allows) | seconds–minutes | low |
| Retrieval params (top_k, weights) | config revert | seconds | low |
| Code | redeploy previous image | minutes | medium |
| Chunking config | **reindex** into a new collection, then flip alias | hours | medium |
| Embedding model | **reindex** (different vector space — mixing is silently wrong) | hours | **high** |
| Corpus document | serve previous `doc@version` | seconds | low |

**Why the embedding model is special:** vectors from two models are not
comparable. Writing new vectors into an existing collection produces
retrieval that is not broken loudly but *degraded quietly* — the worst
failure mode. Hence: new collection, dual-write or backfill, evaluate,
flip the alias, keep the old collection until the next release is proven.

---

## 4. Index versioning and the alias flip

```text
docs_v6  (bge-m3,  chunk 400/60)  ← alias `docs_current` (serving)
docs_v7  (bge-m3,  chunk 512/64)  ← building, then evaluated

build → evaluate offline against dataset@version → compare to baseline
      → gate passes → flip alias docs_current → docs_v7
      → keep docs_v6 for N days → rollback = flip back
```

The alias is the only thing the application knows; it never names a
collection directly. `index_version` is a hash of its inputs, so an index
can never be confused about which embedding model or chunking produced it.

---

## 5. Prompt registry

```text
prompts/rag/rag_answer/
  v3.1.0.md          # immutable once released
  v3.2.0.md
  meta.yaml          # owner, purpose, model compatibility, changelog,
                     # eval results per version, current stage pointers
```

Rules:
- files are immutable after release — a change is a new version, never an edit
- semver: patch = wording, minor = behaviour, **major = output contract change**
- `meta.yaml` carries `stages: {dev: 3.3.0-rc1, staging: 3.2.0, production: 3.2.0}`
- every prompt version records the eval run that justified its promotion
- model compatibility is declared (a prompt tuned for a 7B may regress on a 1.5B)
- CI fails if a released version file is modified (hash check)

Model registry (`models/registry.yaml`) mirrors this shape: name, quant,
revision, weights hash, context length, VRAM estimate (from
`docs/metrics-and-capacity.md` §5), stage, eval results.

---

## 6. Data versioning

```text
Postgres: documents(doc_id, version, content_sha, classification, effective_date,
                    superseded_by, uploaded_by, uploaded_at)
Object store: raw/{doc_id}/{version}/{filename}
Qdrant payload: {doc_id, doc_version, chunk_id, index_version, acl, page}
```

Consequences:
- a citation is `(doc_id, doc_version, chunk_id)` — reproducible forever
- re-uploading creates a new version; the old one remains queryable by
  auditors and remains the correct answer for "what did the Q2 report say
  before the restatement?"
- corpus snapshots (`corpus_snapshot: 2026-09-23`) let an index be rebuilt
  byte-identically

Eval datasets are versioned the same way: a metric is meaningless without
the dataset version it was measured on, so MLflow runs pin both.

---

## 7. Reproducing "production from yesterday"

```bash
# 1. resolve
release=$(cat releases/2026-09-23T*.yaml | yq .release_id)
# 2. code + images
git checkout <git_sha>; docker pull llmops/llm-application:<tag>
# 3. artifacts by pointer
scripts/pin.py --release releases/2026-09-23T18-02-00Z.yaml
#    → sets prompt/model/agent pointers, retrieval config, and
#      docs_current alias → the collection named in the manifest
# 4. verify
uv run poe eval --dataset core_qa@1.6.0 --expect releases/.../quality_gate
```

If the replayed evaluation does not reproduce the recorded metrics, one of
the nine classes is unversioned or a pointer leaked — that is a platform
bug, and the gap analysis is the exercise (Phase 35).

---

## 8. What CI enforces (Phase 34)

- released prompt/skill files are immutable (content hash check)
- any change to embedding model or chunking config **must** bump
  `index_version` and cannot be deployed without a reindex job id
- a release manifest is generated on every deploy; deploy fails if any
  artifact is unpinned or resolves to `latest`
- `latest` is banned for images and model revisions
- the quality gate from `docs/business-requirements.md` §4 is evaluated
  against the pinned dataset version before promotion
