"""Prompt registry: immutable released versions, mutable stage pointers.

Layout:

    prompts/<family>/<name>/
        meta.yaml          stages, owner, changelog, per-version metadata
        v1.0.0.md          immutable once released
        v1.1.0.md

Two rules do the work.

**Versions are immutable.** A released `vN.M.P.md` is content-hashed into
`meta.yaml` at release time, and `verify()` fails if the file has changed since.
Editing a released prompt in place is the prompt equivalent of force-pushing
over a tag: every evaluation result that referenced it becomes a lie.

**Stage pointers are mutable.** `production: 1.1.0` can move; `v1.1.0.md` cannot.
That is what makes rollback a pointer move rather than a code change
(ADR-015 artifact class 2).
"""

from __future__ import annotations

import hashlib
import re
from datetime import UTC, datetime
from enum import StrEnum
from pathlib import Path
from string import Formatter
from typing import Any

import yaml
from pydantic import BaseModel, Field

from llmops_core.errors import ApiError, NotFoundError

PROMPTS_ROOT = Path(__file__).resolve().parents[4] / "prompts"
_SEMVER = re.compile(r"^(\d+)\.(\d+)\.(\d+)$")


class Stage(StrEnum):
    DEV = "dev"
    STAGING = "staging"
    PRODUCTION = "production"


class ImmutabilityError(ApiError):
    """A released prompt file changed on disk. CI treats this as a failed build."""

    status_code = 409
    code = "prompt_immutability_violation"


class PromptVersion(BaseModel):
    version: str = Field(pattern=r"^\d+\.\d+\.\d+$")
    sha256: str
    released_at: str
    # A prompt tuned for a 7B model can regress on a 1.5B one (CLAUDE.md §30),
    # so compatibility is declared rather than assumed.
    model_compatibility: list[str] = []
    variables: list[str] = []
    eval_run: str | None = None
    notes: str = ""


class PromptMeta(BaseModel):
    name: str
    family: str
    purpose: str
    owner: str
    stages: dict[str, str] = {}
    versions: list[PromptVersion] = []

    def version(self, v: str) -> PromptVersion:
        for pv in self.versions:
            if pv.version == v:
                return pv
        raise NotFoundError(f"prompt {self.family}/{self.name} has no version {v}")

    def resolve(self, ref: str) -> str:
        """Accepts a stage name or an explicit version. Stages are indirection on
        purpose: callers ask for `production`, and rollback changes what that means
        without touching the caller."""
        if _SEMVER.match(ref):
            return ref
        if ref in self.stages:
            return self.stages[ref]
        raise NotFoundError(
            f"prompt {self.family}/{self.name}: '{ref}' is neither a version nor a "
            f"configured stage (have: {sorted(self.stages)})"
        )


class Prompt(BaseModel):
    """A resolved prompt: the text plus everything needed to attribute an answer."""

    family: str
    name: str
    version: str
    sha256: str
    template: str
    variables: list[str]
    model_compatibility: list[str]

    @property
    def ref(self) -> str:
        return f"{self.family}/{self.name}@{self.version}"

    def render(self, **values: Any) -> str:
        """Fails on a missing or unexpected variable rather than rendering a prompt
        with a literal `{context}` in it, which a model will happily answer."""
        provided = set(values)
        declared = set(self.variables)
        if missing := declared - provided:
            raise ApiError(f"{self.ref}: missing variables {sorted(missing)}")
        if extra := provided - declared:
            raise ApiError(f"{self.ref}: undeclared variables {sorted(extra)}")
        return self.template.format(**values)


def _sha256(text: str) -> str:
    """Hash the normalised text, not the bytes: a CRLF checkout must not look like
    an edited prompt. The project hit exactly this with Dockerfiles in Phase 9."""
    return hashlib.sha256(text.replace("\r\n", "\n").encode("utf-8")).hexdigest()


def declared_variables(template: str) -> list[str]:
    return sorted({f for _, f, _, _ in Formatter().parse(template) if f})


class PromptRegistry:
    def __init__(self, root: Path = PROMPTS_ROOT) -> None:
        self.root = root

    def _dir(self, family: str, name: str) -> Path:
        d = self.root / family / name
        if not (d / "meta.yaml").exists():
            raise NotFoundError(f"no prompt registry at {d}")
        return d

    def meta(self, family: str, name: str) -> PromptMeta:
        d = self._dir(family, name)
        return PromptMeta(**yaml.safe_load((d / "meta.yaml").read_text(encoding="utf-8")))

    def save_meta(self, meta: PromptMeta) -> None:
        d = self.root / meta.family / meta.name
        d.mkdir(parents=True, exist_ok=True)
        (d / "meta.yaml").write_text(
            yaml.safe_dump(meta.model_dump(), sort_keys=False, width=100), encoding="utf-8"
        )

    def get(self, family: str, name: str, ref: str = Stage.PRODUCTION) -> Prompt:
        meta = self.meta(family, name)
        version = meta.resolve(str(ref))
        pv = meta.version(version)
        path = self._dir(family, name) / f"v{version}.md"
        if not path.exists():
            raise NotFoundError(f"{meta.family}/{meta.name}@{version}: {path} is missing")

        text = path.read_text(encoding="utf-8")
        actual = _sha256(text)
        if actual != pv.sha256:
            raise ImmutabilityError(
                f"{meta.family}/{meta.name}@{version} has been edited since release: "
                f"recorded {pv.sha256[:12]}, on disk {actual[:12]}. "
                f"Released versions are immutable — add a new version instead."
            )
        return Prompt(
            family=meta.family,
            name=meta.name,
            version=version,
            sha256=actual,
            template=text,
            variables=pv.variables,
            model_compatibility=pv.model_compatibility,
        )

    def release(
        self,
        family: str,
        name: str,
        version: str,
        *,
        model_compatibility: list[str] | None = None,
        notes: str = "",
        stage: Stage | None = None,
    ) -> PromptVersion:
        """Record a new version's hash. Refuses to re-release an existing version:
        that is the mutation this registry exists to prevent."""
        if not _SEMVER.match(version):
            raise ApiError(f"version must be semver, got {version!r}")
        d = self._dir(family, name)
        path = d / f"v{version}.md"
        if not path.exists():
            raise NotFoundError(f"write the prompt to {path} before releasing it")

        meta = self.meta(family, name)
        if any(pv.version == version for pv in meta.versions):
            raise ImmutabilityError(
                f"{family}/{name}@{version} is already released. Bump the version "
                f"instead of re-releasing — patch for wording, minor for behaviour, "
                f"major for an output-contract change."
            )

        text = path.read_text(encoding="utf-8")
        pv = PromptVersion(
            version=version,
            sha256=_sha256(text),
            released_at=datetime.now(UTC).isoformat(timespec="seconds"),
            model_compatibility=model_compatibility or [],
            variables=declared_variables(text),
            notes=notes,
        )
        meta.versions.append(pv)
        meta.versions.sort(key=lambda x: tuple(int(p) for p in x.version.split(".")))
        if stage:
            meta.stages[str(stage)] = version
        self.save_meta(meta)
        return pv

    def promote(self, family: str, name: str, version: str, stage: Stage) -> tuple[str | None, str]:
        """Move a stage pointer. Returns (previous, new) so the caller can log the
        transition — a promotion with no record is indistinguishable from a drift."""
        meta = self.meta(family, name)
        meta.version(version)  # raises if the version was never released
        previous = meta.stages.get(str(stage))
        meta.stages[str(stage)] = version
        self.save_meta(meta)
        return previous, version

    def rollback(self, family: str, name: str, stage: Stage = Stage.PRODUCTION) -> tuple[str, str]:
        """Point a stage at the previously released version. The cheapest rollback in
        the nine artifact classes: a pointer move, seconds, no reindex."""
        meta = self.meta(family, name)
        current = meta.stages.get(str(stage))
        if current is None:
            raise NotFoundError(f"{family}/{name} has no {stage} pointer to roll back")
        ordered = [pv.version for pv in meta.versions]
        idx = ordered.index(current)
        if idx == 0:
            raise ApiError(
                f"{family}/{name}@{current} is the earliest version; nothing to roll back to"
            )
        target = ordered[idx - 1]
        meta.stages[str(stage)] = target
        self.save_meta(meta)
        return current, target

    def list_prompts(self) -> list[tuple[str, str]]:
        if not self.root.exists():
            return []
        return sorted(
            (p.parent.parent.name, p.parent.name) for p in self.root.glob("*/*/meta.yaml")
        )

    def verify(self) -> list[str]:
        """Every released version, hashed against its file. This is the CI gate:
        a non-empty return means a released prompt was edited in place."""
        problems: list[str] = []
        for family, name in self.list_prompts():
            meta = self.meta(family, name)
            d = self.root / family / name
            for pv in meta.versions:
                path = d / f"v{pv.version}.md"
                if not path.exists():
                    problems.append(f"{family}/{name}@{pv.version}: file missing ({path.name})")
                    continue
                actual = _sha256(path.read_text(encoding="utf-8"))
                if actual != pv.sha256:
                    problems.append(
                        f"{family}/{name}@{pv.version}: edited after release "
                        f"(recorded {pv.sha256[:12]}, on disk {actual[:12]})"
                    )
            for stage, version in meta.stages.items():
                if not any(pv.version == version for pv in meta.versions):
                    problems.append(
                        f"{family}/{name}: stage {stage} points at {version}, never released"
                    )
            # An orphan file is a released-looking prompt nothing can resolve.
            released = {f"v{pv.version}.md" for pv in meta.versions}
            for path in d.glob("v*.md"):
                if path.name not in released:
                    problems.append(f"{family}/{name}: {path.name} exists but is not released")
        return problems
