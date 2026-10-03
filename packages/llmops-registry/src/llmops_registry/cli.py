"""Registry CLI.

    uv run llmops-registry verify                      # the CI gate
    uv run llmops-registry prompts
    uv run llmops-registry show rag/rag_answer
    uv run llmops-registry release rag/rag_answer 1.1.0 --stage dev --compat qwen2.5-7b
    uv run llmops-registry promote rag/rag_answer 1.1.0 --stage production
    uv run llmops-registry rollback rag/rag_answer --stage production
    uv run llmops-registry models
    uv run llmops-registry promote-model Qwen2.5-7B-Instruct-AWQ --stage staging

`verify` exits non-zero on any violation, which is how CI blocks an edited
released prompt or a model registry that cannot physically fit the card.
"""

from __future__ import annotations

import argparse
import sys

from llmops_registry.models import ModelRegistry
from llmops_registry.models import Stage as ModelStage
from llmops_registry.prompts import PromptRegistry
from llmops_registry.prompts import Stage as PromptStage


def _split(ref: str) -> tuple[str, str]:
    if "/" not in ref:
        raise SystemExit(f"expected family/name, got {ref!r}")
    family, name = ref.split("/", 1)
    return family, name


def cmd_verify(_: argparse.Namespace) -> int:
    prompts = PromptRegistry().verify()
    models = ModelRegistry().verify()
    if not prompts and not models:
        print("registry OK — all released prompts match their recorded hashes,")
        print("             model stages are unique and production models fit the card")
        return 0
    for p in prompts:
        print(f"PROMPT  {p}")
    for m in models:
        print(f"MODEL   {m}")
    print(f"\n{len(prompts) + len(models)} violation(s)")
    return 1


def cmd_prompts(_: argparse.Namespace) -> int:
    reg = PromptRegistry()
    entries = reg.list_prompts()
    if not entries:
        print("no prompts registered")
        return 0
    print(f"  {'prompt':<28}{'versions':>10}  stages")
    for family, name in entries:
        meta = reg.meta(family, name)
        stages = ", ".join(f"{k}={v}" for k, v in sorted(meta.stages.items())) or "-"
        print(f"  {f'{family}/{name}':<28}{len(meta.versions):>10}  {stages}")
    return 0


def cmd_show(args: argparse.Namespace) -> int:
    family, name = _split(args.ref)
    reg = PromptRegistry()
    meta = reg.meta(family, name)
    print(f"{meta.family}/{meta.name}")
    print(f"  purpose  {meta.purpose}")
    print(f"  owner    {meta.owner}")
    print(f"  stages   {dict(sorted(meta.stages.items())) or '-'}")
    print("  versions")
    for pv in meta.versions:
        at = [s for s, v in meta.stages.items() if v == pv.version]
        marker = f"  <- {', '.join(sorted(at))}" if at else ""
        print(f"    {pv.version:<9} {pv.sha256[:12]}  {pv.released_at}{marker}")
        if pv.variables:
            print(f"      variables   {pv.variables}")
        if pv.model_compatibility:
            print(f"      compatible  {pv.model_compatibility}")
        if pv.notes:
            print(f"      notes       {pv.notes}")
    return 0


def cmd_release(args: argparse.Namespace) -> int:
    family, name = _split(args.ref)
    pv = PromptRegistry().release(
        family,
        name,
        args.version,
        model_compatibility=args.compat or [],
        notes=args.notes or "",
        stage=PromptStage(args.stage) if args.stage else None,
    )
    print(f"released {family}/{name}@{pv.version}  sha256 {pv.sha256[:12]}")
    print(f"  variables {pv.variables}")
    if args.stage:
        print(f"  stage {args.stage} -> {pv.version}")
    return 0


def cmd_promote(args: argparse.Namespace) -> int:
    family, name = _split(args.ref)
    previous, new = PromptRegistry().promote(family, name, args.version, PromptStage(args.stage))
    print(f"{family}/{name}: {args.stage} {previous or '(unset)'} -> {new}")
    return 0


def cmd_rollback(args: argparse.Namespace) -> int:
    family, name = _split(args.ref)
    was, now = PromptRegistry().rollback(family, name, PromptStage(args.stage))
    print(f"{family}/{name}: {args.stage} rolled back {was} -> {now}")
    print("  pointer moved only; no file changed, no reindex needed")
    return 0


def cmd_models(_: argparse.Namespace) -> int:
    entries = ModelRegistry().load()
    print(f"  {'model':<34}{'kind':<12}{'stage':<11}{'revision':<12}{'VRAM GB':>8}")
    for e in entries:
        vram = f"{e.vram_estimate_gb:.2f}" if e.vram_estimate_gb is not None else "-"
        print(f"  {e.name:<34}{e.kind:<12}{e.stage:<11}{e.revision[:10]:<12}{vram:>8}")
    return 0


def cmd_promote_model(args: argparse.Namespace) -> int:
    previous, new = ModelRegistry().promote(args.name, ModelStage(args.stage))
    print(f"{args.name}: {previous} -> {new}")
    return 0


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(prog="llmops-registry", description=__doc__)
    sub = parser.add_subparsers(dest="cmd", required=True)

    sub.add_parser("verify", help="CI gate: hashes, stages, VRAM fit").set_defaults(fn=cmd_verify)
    sub.add_parser("prompts", help="list registered prompts").set_defaults(fn=cmd_prompts)
    sub.add_parser("models", help="list registered models").set_defaults(fn=cmd_models)

    p = sub.add_parser("show", help="show one prompt's versions and stages")
    p.add_argument("ref", help="family/name")
    p.set_defaults(fn=cmd_show)

    p = sub.add_parser("release", help="record a new version's hash")
    p.add_argument("ref")
    p.add_argument("version")
    p.add_argument("--stage", choices=[str(s) for s in PromptStage])
    p.add_argument("--compat", action="append", help="compatible model (repeatable)")
    p.add_argument("--notes", default="")
    p.set_defaults(fn=cmd_release)

    p = sub.add_parser("promote", help="move a stage pointer to a released version")
    p.add_argument("ref")
    p.add_argument("version")
    p.add_argument(
        "--stage", default=str(PromptStage.PRODUCTION), choices=[str(s) for s in PromptStage]
    )
    p.set_defaults(fn=cmd_promote)

    p = sub.add_parser("rollback", help="point a stage at the previous version")
    p.add_argument("ref")
    p.add_argument(
        "--stage", default=str(PromptStage.PRODUCTION), choices=[str(s) for s in PromptStage]
    )
    p.set_defaults(fn=cmd_rollback)

    p = sub.add_parser("promote-model", help="move a model to a stage")
    p.add_argument("name")
    p.add_argument("--stage", required=True, choices=[str(s) for s in ModelStage])
    p.set_defaults(fn=cmd_promote_model)

    args = parser.parse_args(argv)
    return int(args.fn(args))


if __name__ == "__main__":
    sys.exit(main())
