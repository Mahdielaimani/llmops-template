# ADR-021: Prompt registry — files in Git, hashes for immutability, pointers for stages

**Status:** Accepted
**Date:** 2026-10-03
**Phase:** 10

## Context
`docs/versioning.md` requires prompts to be versioned artifact class 2:
immutable once released, independently rollable-back, and stamped on every
response. Phase 15 will load them at runtime; Phase 12 needs a judge prompt;
Phase 13 needs to attach evaluation runs to specific versions.

The failure mode being designed against is specific. A prompt edited in place
silently invalidates every evaluation result that referenced it — there is no way
afterwards to know which text produced which score. And because a prompt is
executable instruction text, an edit is also an **injection vector**: appending
"ignore all previous rules" to the production prompt is a complete bypass of
whatever the prompt was enforcing.

## Decision
Prompts are **Markdown files in Git**, one directory per prompt, with a
`meta.yaml` recording a content hash per released version and a mutable map of
stage pointers.

```text
prompts/<family>/<name>/
    meta.yaml       stages (mutable) + versions (immutable, hashed)
    v1.0.0.md
```

1. **Versions immutable, stages mutable.** Releasing records
   `sha256(newline-normalised text)`. `verify()` recomputes and fails on
   mismatch. `promote` and `rollback` only move pointers.
2. **Hash the normalised text, not the bytes.** A CRLF checkout must not look
   like an edit — Phase 9 lost time to exactly that with Dockerfiles.
3. **Variables are derived from the template, not declared by hand**, and
   `render()` refuses both a missing and an *undeclared* variable. A renamed
   placeholder then fails at startup rather than shipping a prompt containing a
   literal `{context}`, which a model will answer anyway.
4. **Model compatibility is declared per version** (CLAUDE.md §30).
5. **`verify` is part of `poe check`**, so a bad release breaks the local build,
   not only CI.

## Alternatives considered
| Option | Why not |
| --- | --- |
| **Prompts as string literals in Python** | the default, and the worst option: a prompt change becomes a code deploy, there is no version to attach an eval result to, and diffs are buried in application commits |
| **MLflow Prompt Registry** | genuinely good: versioning, aliases and lineage to runs, which is most of this ADR. Rejected for now because it puts prompts behind a running service, so the application cannot start without MLflow and prompts stop being reviewable in a pull request. Revisit at Phase 13, where MLflow arrives anyway |
| **LangSmith / Langfuse prompt management** | strong UI, non-engineer editing, A/B support. SaaS dependency, prompts leave the repo, and prompt review stops being code review |
| **Database table** | easy to mutate, which is the problem; and a prompt change should be a reviewed diff |
| **Git tags per prompt version** | immutability for free, but one tag per prompt version pollutes the tag namespace and cannot express per-prompt stage pointers |
| **Content-addressed filenames** (`prompt-5eee00f5.md`) | strictly immutable, and unreadable. Semver plus a recorded hash gives the same guarantee with names a human can discuss |
| **Trust review instead of hashing** | review catches intentional edits by cooperative people; the hash catches careless ones, bad merges, and the injection case |

## Trade-offs
+ a prompt change is a reviewed diff, with its own version and changelog
+ rollback is a pointer move, seconds, no deploy, both files retained
+ the immutability check doubles as a security control for prompt tampering
+ no runtime dependency: prompts load from the filesystem, so the app starts offline
+ `verify` fails the build on six distinct violation classes
− **two sources of truth once MLflow arrives** in Phase 13: prompt text in Git,
  eval runs in MLflow, joined by the `eval_run` field. Workable, not elegant
− non-engineers cannot edit a prompt without a pull request. For a finance
  domain where prompt wording is a controls question, that is arguably correct —
  but it is a real limitation if prompt iteration ever needs to be fast
− `meta.yaml` is hand-maintained by the CLI, so a manual edit can desynchronise
  it from the files. `verify` catches every case we could enumerate
− semver on prose is a judgement call; the patch/minor/major rule is written down
  but cannot be mechanically enforced

## Consequences
- `prompts/` is the source of truth; Phase 15 resolves `production` at startup
  and stamps `prompt_version` on every response (ADR-017).
- Phase 13 populates `eval_run` so a promotion can cite the evidence for it.
- Phase 34 runs `llmops-registry verify` on every pull request.
- Phase 35's release manifest pins the resolved prompt version plus its hash, so
  "reproduce yesterday" resolves to exact text.
- A major version bump is an output-contract change and needs a coordinated
  release with whatever parses the output.
- The same registry pattern covers models (`models/registry.yaml`), where
  `verify` additionally rejects moving revisions, duplicate stages, and a
  production model that cannot physically fit the card.
