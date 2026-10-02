# ADR-011: Monorepo with uv workspace + poethepoet task runner

**Status:** Accepted
**Date:** 2026-09-17
**Phase:** 1

## Context
The platform is many deployables (5 apps, 4 services, workers, eval jobs)
that share primitives and must pin identical dependency versions to avoid
"works in rag-service, breaks in agent-service". The host is Windows with
no `make`; `uv` 0.8 is installed; Python 3.14 is the system default but
ML wheels target 3.12.

## Decision
- Single Git repo. `packages/*` (shared libs), `apps/*`, `services/*` are
  uv **workspace members**; one `uv.lock`, one `.venv`, Python pinned to
  3.12 via `.python-version`.
- Each member has its own `pyproject.toml` and Dockerfile; images are built
  with `uv sync --package <name>` so an image only contains what it needs.
- Tasks are defined once in `[tool.poe.tasks]` and run with `uv run poe
  <task>`; a two-line `Makefile` delegates to the same names for Linux/CI.
- Deviation from the master-prompt layout: added `packages/` for shared
  code (the prompt had no home for it) and `infrastructure/kong/`.

## Alternatives considered
| Option | Why not |
| --- | --- |
| Polyrepo | version drift across services; 10× CI config; wrong for a lab |
| pip + requirements.txt per service | no lock across members, slow, no workspace concept |
| Poetry | slower resolver, workspace support immature vs uv |
| `just` | needs a separate install on every machine/CI image |
| GNU Make only | not on Windows without extra install; tab-sensitive; `make` stays as a shim |
| `uv run` scripts in `[project.scripts]` | no task composition (`check = lint + typecheck + test`) |

## Trade-offs
+ one lockfile, one command surface, image per service, fast installs
− a change in `llmops-core` touches every service (mitigated by tests + CI)
− Docker build context is the repo root (needed for workspace); `.dockerignore` keeps it small

## Consequences
- `uv.lock` is committed; CI uses `uv sync --frozen`.
- New service = new directory + `pyproject.toml` + add to `[tool.uv.sources]`.
- Task names are the contract: `lint fmt typecheck test test-unit test-integration cov check dev up down logs ps`.
