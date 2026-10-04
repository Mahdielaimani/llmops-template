"""Evaluation CLI.

    uv run llmops-eval run --mode hybrid_weighted
    uv run llmops-eval compare-modes            # all five, the Phase 9 re-test
    uv run llmops-eval baseline --mode hybrid_weighted
    uv run llmops-eval gate --mode hybrid_weighted   # exits non-zero on failure

`gate` is what CI calls. It fails on any unauthorized retrieval (Q-6) regardless
of the other numbers, and on Recall@5 or MRR below the thresholds in
docs/business-requirements.md §4.
"""

from __future__ import annotations

import argparse
import json
import os
import shutil
import sys
from typing import Any

from llmops_eval.runner import (
    RESULTS_DIR,
    RunConfig,
    baseline_path,
    build_pipeline,
    compare,
    load_baseline,
    load_dataset,
    run,
    save,
)
from rag_service.pipeline import RetrievalPipeline

QDRANT = os.getenv("LLMOPS_QDRANT__URL", "http://localhost:6333")
MODES = ["dense", "lexical", "hybrid", "hybrid_weighted", "hybrid_rerank"]


def _config(
    mode: str, args: argparse.Namespace
) -> tuple[RunConfig, tuple[RetrievalPipeline, list[dict[str, Any]]]]:
    items, version, sha = load_dataset()
    pipeline, index_version, embed_model, reranker = build_pipeline(QDRANT)
    cfg = RunConfig(
        mode=mode,
        top_k=args.top_k,
        candidates=args.candidates,
        dataset_version=version,
        dataset_sha=sha,
        index_version=index_version,
        embed_model=embed_model,
        reranker=reranker,
    )
    return cfg, (pipeline, items)


def _print_metrics(label: str, metrics: dict[str, Any]) -> None:
    print(f"\n{label}")
    for key in (
        "n",
        "n_answerable",
        "n_unanswerable",
        "recall_at_1",
        "recall_at_3",
        "recall_at_5",
        "recall_at_10",
        "mrr",
        "ndcg_at_5",
        "context_precision",
        "context_recall",
        "acl_violations",
        "missed_count",
    ):
        print(f"  {key:<22}{metrics[key]}")


def cmd_run(args: argparse.Namespace) -> int:
    cfg, (pipeline, items) = _config(args.mode, args)
    result = run(pipeline, items, cfg, check_acl=not args.skip_acl)
    path = save(result)

    print(f"run        {result.run_id}")
    print(f"dataset    {cfg.dataset_version} ({cfg.dataset_sha})  n={len(items)}")
    print(f"index      {cfg.index_version}  embed={cfg.embed_model}")
    print(f"mode       {cfg.mode}  top_k={cfg.top_k}  candidates={cfg.candidates}")
    _print_metrics("metrics", result.metrics)

    print("\nby hardness class")
    print(f"  {'class':<16}{'n':>5}{'R@1':>8}{'R@5':>8}{'MRR':>8}{'ctx prec':>10}")
    for name, m in result.per_hardness.items():
        if not m["n_answerable"]:
            # No document to retrieve, so retrieval metrics are undefined rather
            # than zero. Abstention is scored from Phase 15.
            print(f"  {name:<16}{m['n']:>5}{'n/a':>8}{'n/a':>8}{'n/a':>8}{'n/a':>10}")
            continue
        print(
            f"  {name:<16}{m['n']:>5}{m['recall_at_1']:>8.3f}"
            f"{m['recall_at_5']:>8.3f}{m['mrr']:>8.3f}{m['context_precision']:>10.3f}"
        )

    if result.acl_by_identity:
        print("\nACL violations per identity (gate Q-6 requires 0 everywhere)")
        for name, count in result.acl_by_identity.items():
            print(f"  {name:<18}{count}")

    print(f"\nlatency ms {result.latency_ms}")
    if baseline := load_baseline():
        lines = compare(result.metrics, baseline.metrics)
        print(f"\nvs baseline {baseline.run_id}")
        for line in lines or ["  no metric moved by more than 0.005"]:
            print(f"  {line}" if lines else line)

    if result.gate_failures:
        print("\nGATE FAILURES")
        for f in result.gate_failures:
            print(f"  {f}")
    else:
        print("\ngates passed")
    print(f"\nsaved {path.relative_to(path.parents[1])}")
    return 0


def cmd_compare_modes(args: argparse.Namespace) -> int:
    """Phase 9 compared five configurations on 12 questions and recorded that the
    differences were one or two questions. This re-runs it on the full set, which
    is the only way those numbers decide anything."""
    items, version, sha = load_dataset()
    pipeline, index_version, embed_model, reranker = build_pipeline(QDRANT)
    print(f"dataset {version} ({sha})  n={len(items)}   index {index_version}")

    results = {}
    for mode in MODES:
        cfg = RunConfig(
            mode=mode,
            top_k=args.top_k,
            candidates=args.candidates,
            dataset_version=version,
            dataset_sha=sha,
            index_version=index_version,
            embed_model=embed_model,
            reranker=reranker,
        )
        results[mode] = run(pipeline, items, cfg, check_acl=False)

    print(
        f"\n  {'mode':<18}{'R@1':>8}{'R@3':>8}{'R@5':>8}{'MRR':>8}"
        f"{'nDCG@5':>9}{'ctxP':>8}{'p50 ms':>9}"
    )
    for mode, r in results.items():
        m = r.metrics
        print(
            f"  {mode:<18}{m['recall_at_1']:>8.3f}{m['recall_at_3']:>8.3f}"
            f"{m['recall_at_5']:>8.3f}{m['mrr']:>8.3f}{m['ndcg_at_5']:>9.3f}"
            f"{m['context_precision']:>8.3f}{r.latency_ms.get('p50', 0):>9.1f}"
        )

    classes = sorted({c for r in results.values() for c in r.per_hardness})
    print(f"\nMRR by hardness class\n  {'class':<16}" + "".join(f"{m[:13]:>15}" for m in MODES))
    for cls in classes:
        row = f"  {cls:<16}"
        for mode in MODES:
            cls_metrics: dict[str, Any] | None = results[mode].per_hardness.get(cls)
            if cls_metrics is None:
                row += f"{'-':>15}"
            elif not cls_metrics["n_answerable"]:
                row += f"{'n/a':>15}"
            else:
                row += f"{cls_metrics['mrr']:>15.3f}"
        print(row)

    best = max(results.items(), key=lambda kv: float(kv[1].metrics["mrr"]))
    n = len(items)
    print(f"\nbest by MRR: {best[0]} ({best[1].metrics['mrr']:.3f})")
    print(f"one question is worth {1 / n:.4f} MRR at n={n}, against 0.0833 at Phase 9's n=12")
    for r in results.values():
        save(r)
    return 0


def cmd_baseline(args: argparse.Namespace) -> int:
    cfg, (pipeline, items) = _config(args.mode, args)
    result = run(pipeline, items, cfg)
    if result.gate_failures:
        print("refusing to set a baseline from a run that fails its gates:")
        for f in result.gate_failures:
            print(f"  {f}")
        return 1
    path = save(result)
    shutil.copyfile(path, baseline_path())
    print(f"baseline set from {result.run_id}")
    print(
        f"  recall@5 {result.metrics['recall_at_5']}  mrr {result.metrics['mrr']}  "
        f"acl_violations {result.metrics['acl_violations']}"
    )
    return 0


def cmd_gate(args: argparse.Namespace) -> int:
    cfg, (pipeline, items) = _config(args.mode, args)
    result = run(pipeline, items, cfg)
    if result.gate_failures:
        for f in result.gate_failures:
            print(f"FAIL  {f}")
        return 1
    print(
        f"PASS  recall@5 {result.metrics['recall_at_5']}  mrr {result.metrics['mrr']}  "
        f"acl_violations 0  (dataset {cfg.dataset_version}/{cfg.dataset_sha})"
    )
    return 0


def cmd_show_baseline(_: argparse.Namespace) -> int:
    baseline = load_baseline()
    if baseline is None:
        print(f"no baseline at {baseline_path()}")
        return 1
    print(json.dumps({"run_id": baseline.run_id, **baseline.metrics}, indent=2))
    return 0


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(prog="llmops-eval", description=__doc__)
    parser.add_argument("--results", default=str(RESULTS_DIR), help=argparse.SUPPRESS)
    sub = parser.add_subparsers(dest="cmd", required=True)

    def common(p: argparse.ArgumentParser) -> None:
        p.add_argument("--mode", default="hybrid_weighted", choices=MODES)
        p.add_argument("--top-k", type=int, default=5)
        p.add_argument("--candidates", type=int, default=20)

    p = sub.add_parser("run", help="evaluate one configuration")
    common(p)
    p.add_argument("--skip-acl", action="store_true", help="skip the per-identity ACL sweep")
    p.set_defaults(fn=cmd_run)

    p = sub.add_parser("compare-modes", help="all five configurations on the full set")
    common(p)
    p.set_defaults(fn=cmd_compare_modes)

    p = sub.add_parser("baseline", help="set the comparison baseline (refuses a failing run)")
    common(p)
    p.set_defaults(fn=cmd_baseline)

    p = sub.add_parser("gate", help="CI gate; exits non-zero on failure")
    common(p)
    p.set_defaults(fn=cmd_gate)

    sub.add_parser("show-baseline", help="print the current baseline").set_defaults(
        fn=cmd_show_baseline
    )

    args = parser.parse_args(argv)
    return int(args.fn(args))


if __name__ == "__main__":
    sys.exit(main())
