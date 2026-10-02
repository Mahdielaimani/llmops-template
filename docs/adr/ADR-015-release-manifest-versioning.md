# ADR-015: Nine versioned artifact classes and a release manifest

**Status:** Accepted
**Date:** 2026-09-24
**Phase:** 10 (partial) → 35 (complete)

## Context
In a classical service, "the deployed version" is one image tag. In an LLM
system the answer can change without any code change: a prompt edit, a
model revision, a different embedding model, a re-chunked corpus, a new
document version, a retrieval parameter. Debugging "quality dropped
yesterday" is impossible unless every one of those is pinned and stamped.

The regression exercise (Phase 14) and the rollback exercise (Phase 35)
both require answering: *what exactly was serving at 18:02 yesterday?*

## Decision
Version nine artifact classes independently — code, prompt, model,
embedding model, retrieval/chunking config, corpus document, index, eval
dataset, agent/skill — each with its own key, system of record and
rollback path (table in `docs/versioning.md` §1).

Pin all nine, plus a config hash, into an immutable **release manifest**
(`releases/<timestamp>.yaml`) carrying the quality-gate metrics that
justified the promotion. `release_id` is stamped on every response and log
line, alongside the individual `model_version`, `prompt_version` and
`index_version` used as metric labels.

Breaking-change rule: changing the **embedding model or chunking config**
invalidates the index. Such a change must produce a new `index_version`,
build a new Qdrant collection, pass evaluation, then **flip an alias**.
The application only ever knows the alias. Rollback = flip back.

`latest` is banned for images and model revisions; CI fails a deploy if
any artifact is unpinned.

## Alternatives considered
| Option | Why not |
| --- | --- |
| Git sha alone | prompts/models/data/index can all move without a code change |
| MLflow model registry only | covers models and runs; no home for corpus versions, index identity, agent/skill definitions |
| DVC / LakeFS for everything | strong for data lineage, heavy for a laptop lab; we take the *idea* (content-addressed snapshots) without the dependency |
| Version only what changed "often" | the silent failures come from the rare changes (embedding model), exactly the ones people skip |
| Mutable prompt files edited in place | no way to reproduce or roll back; CI now blocks it by hash |

## Trade-offs
+ "reproduce production from yesterday" becomes a command, not an archaeology project
+ each class rolls back independently, cheapest-first (prompt = seconds, embedding = reindex)
+ metric labels by version turn "quality dropped" into "quality dropped for prompt@3.2.0"
− ceremony: a prompt tweak now means a version bump and an eval run
− storage: old collections and document versions are kept for the rollback window
− manifest generation is one more CI step that can itself break

## Consequences
- Prompt and skill files are **immutable after release**; CI verifies hashes.
- Qdrant is addressed by alias only; no code names a collection directly.
- A citation is `(doc_id, doc_version, chunk_id)` — reproducible forever,
  which is also the auditor requirement in `docs/business-requirements.md`.
- Evaluation results are meaningless without `dataset@version`; MLflow runs
  pin both dataset and manifest.
- Phase 35 acceptance test: replay a manifest and reproduce its recorded
  metrics. A mismatch means something is unversioned — that gap analysis
  is the lesson.
