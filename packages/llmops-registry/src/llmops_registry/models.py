"""Model registry: one file, every model the platform can serve.

`models/registry.yaml` is the answer to "what exactly is running". Each entry
carries what makes a model *reproducible* — a pinned revision and a weights hash
— and what makes it *operable*: VRAM estimate, context limit, stage.

The VRAM estimate is not decoration. `docs/kv-cache.md` §3 showed that
Qwen2.5-7B at fp16 does not fit on this card at all, and the only way to know
that before a deploy is to carry the arithmetic in the registry
(ADR-015 artifact classes 3 and 4).
"""

from __future__ import annotations

from enum import StrEnum
from pathlib import Path
from typing import Any

import yaml
from pydantic import BaseModel, Field

from llmops_core.errors import ApiError, NotFoundError

REGISTRY_PATH = Path(__file__).resolve().parents[4] / "models" / "registry.yaml"


class ModelKind(StrEnum):
    GENERATION = "generation"
    EMBEDDING = "embedding"
    RERANKER = "reranker"
    CLASSICAL = "classical"


class Stage(StrEnum):
    DEV = "dev"
    STAGING = "staging"
    PRODUCTION = "production"
    RETIRED = "retired"


class ModelEntry(BaseModel):
    name: str
    kind: ModelKind
    provider: str
    # `latest` is banned (ADR-015): a revision that moves makes every recorded
    # evaluation result unreproducible.
    revision: str = Field(min_length=1)
    quantization: str | None = None
    params_b: float | None = None
    context_length: int | None = None
    dim: int | None = None
    vram_estimate_gb: float | None = None
    kv_bytes_per_token: int | None = None
    sha256: str | None = None
    stage: Stage = Stage.DEV
    notes: str = ""

    def model_post_init(self, _: Any) -> None:
        if self.revision.strip().lower() in {"latest", "main", "head"}:
            raise ApiError(
                f"model {self.name}: revision {self.revision!r} is a moving target; "
                f"pin an immutable revision (ADR-015)"
            )

    @property
    def ref(self) -> str:
        quant = f":{self.quantization}" if self.quantization else ""
        return f"{self.name}{quant}@{self.revision}"

    def fits_in(self, vram_gb: float, *, overhead_gb: float = 1.2) -> bool:
        """Weights plus engine overhead against the card. Does not include the KV
        cache — that is context-dependent and lives in docs/kv-cache.md §4."""
        if self.vram_estimate_gb is None:
            return True  # unknown, not claimed to fit
        return self.vram_estimate_gb + overhead_gb <= vram_gb


class ModelRegistry:
    def __init__(self, path: Path = REGISTRY_PATH) -> None:
        self.path = path

    def load(self) -> list[ModelEntry]:
        if not self.path.exists():
            raise NotFoundError(f"no model registry at {self.path}")
        raw = yaml.safe_load(self.path.read_text(encoding="utf-8")) or {}
        return [ModelEntry(**entry) for entry in raw.get("models", [])]

    def save(self, entries: list[ModelEntry]) -> None:
        self.path.parent.mkdir(parents=True, exist_ok=True)
        payload = {"models": [e.model_dump(mode="json", exclude_none=True) for e in entries]}
        self.path.write_text(yaml.safe_dump(payload, sort_keys=False, width=100), encoding="utf-8")

    def get(self, name: str) -> ModelEntry:
        for e in self.load():
            if e.name == name:
                return e
        raise NotFoundError(f"model {name!r} is not in {self.path.name}")

    def current(self, kind: ModelKind, stage: Stage = Stage.PRODUCTION) -> ModelEntry:
        """The one model of a kind at a stage. Two models at the same stage is a
        configuration error, not a choice to make at request time."""
        matches = [e for e in self.load() if e.kind is kind and e.stage is stage]
        if not matches:
            raise NotFoundError(f"no {kind} model at stage {stage}")
        if len(matches) > 1:
            raise ApiError(
                f"{len(matches)} {kind} models are at stage {stage} "
                f"({', '.join(m.name for m in matches)}); exactly one must be"
            )
        return matches[0]

    def promote(self, name: str, stage: Stage) -> tuple[Stage, Stage]:
        """Promotion demotes whatever held that stage for the same kind, so the
        invariant in `current()` cannot be broken by a promotion."""
        entries = self.load()
        target = next((e for e in entries if e.name == name), None)
        if target is None:
            raise NotFoundError(f"model {name!r} is not in {self.path.name}")
        previous = target.stage
        if stage in (Stage.PRODUCTION, Stage.STAGING):
            for e in entries:
                if e.kind is target.kind and e.stage is stage and e.name != name:
                    e.stage = Stage.DEV
        target.stage = stage
        self.save(entries)
        return previous, stage

    def verify(self, *, vram_gb: float = 8.0) -> list[str]:
        """CI gate. Catches the three ways this file goes wrong: a moving revision,
        two models claiming the same stage, and a production model that cannot
        physically fit the hardware it is pinned to."""
        problems: list[str] = []
        entries = self.load()

        for kind in ModelKind:
            for stage in (Stage.PRODUCTION, Stage.STAGING):
                at_stage = [e for e in entries if e.kind is kind and e.stage is stage]
                if len(at_stage) > 1:
                    problems.append(
                        f"{len(at_stage)} {kind} models at stage {stage}: "
                        f"{', '.join(e.name for e in at_stage)}"
                    )

        for e in entries:
            if e.stage is Stage.PRODUCTION and not e.fits_in(vram_gb):
                problems.append(
                    f"{e.name} is at production but needs ~{e.vram_estimate_gb} GB of weights "
                    f"plus overhead on a {vram_gb} GB card (docs/kv-cache.md §3)"
                )
            if e.kind is ModelKind.EMBEDDING and e.dim is None:
                problems.append(f"{e.name}: embedding models must declare `dim` (index identity)")
            if e.kind is ModelKind.GENERATION and e.context_length is None:
                problems.append(f"{e.name}: generation models must declare `context_length`")

        names = [e.name for e in entries]
        for name in {n for n in names if names.count(n) > 1}:
            problems.append(f"duplicate model entry: {name}")
        return problems
