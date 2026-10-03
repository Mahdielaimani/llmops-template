"""Compare the four retrieval configurations on the hard question set.

    docker compose --profile core up -d qdrant
    uv run python -m rag_service.corpus && uv run python -m rag_service.ingest
    uv run python benchmarks/retrieval_modes.py

This is the Phase 9 gate. Phase 8 measured Recall@3 = 1.000 with dense retrieval
alone on the planted facts, which is why the hard set exists: a comparison with
no headroom measures noise. Results are broken down by *hardness class* so the
mechanism is visible, not just the aggregate.
"""

from __future__ import annotations

import json
import os
import statistics
import sys
from collections import defaultdict
from pathlib import Path

from qdrant_client import QdrantClient

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "apps" / "rag-service" / "src"))

from rag_service.acl import AclPredicate  # noqa: E402
from rag_service.embedding import Embedder  # noqa: E402
from rag_service.fusion import Reranker  # noqa: E402
from rag_service.lexical import LexicalIndex, load_payloads  # noqa: E402
from rag_service.main import _STUB_USERS  # noqa: E402
from rag_service.pipeline import Mode, RetrievalPipeline  # noqa: E402
from rag_service.retriever import Retriever  # noqa: E402

QDRANT = os.getenv("LLMOPS_QDRANT__URL", "http://localhost:6333")
HARD = ROOT / "data" / "evaluation" / "hard_questions.json"
FACTS = ROOT / "data" / "evaluation" / "planted_facts.json"
MODES = [Mode.DENSE, Mode.LEXICAL, Mode.HYBRID, Mode.HYBRID_WEIGHTED, Mode.HYBRID_RERANK]


def rank_of(chunks: list, doc_id: str, version: int | None) -> int | None:
    for i, c in enumerate(chunks, start=1):
        if c.doc_id == doc_id and (version is None or c.doc_version == version):
            return i
    return None


def score(ranks: list[int | None]) -> dict[str, float]:
    n = len(ranks)
    return {
        "recall@1": sum(1 for r in ranks if r == 1) / n,
        "recall@3": sum(1 for r in ranks if r and r <= 3) / n,
        "recall@5": sum(1 for r in ranks if r and r <= 5) / n,
        "mrr": sum(1 / r for r in ranks if r) / n,
    }


def main() -> int:
    hard = json.loads(HARD.read_text(encoding="utf-8"))
    facts = json.loads(FACTS.read_text(encoding="utf-8"))
    client = QdrantClient(url=QDRANT)
    collections = sorted(
        c.name for c in client.get_collections().collections if c.name.startswith("docs_")
    )
    if not collections:
        print("no docs_* collection; run `python -m rag_service.ingest`", file=sys.stderr)
        return 1
    collection = collections[-1]

    embedder = Embedder()
    payloads = load_payloads(client, collection)
    pipeline = RetrievalPipeline(
        Retriever(client, embedder, collection), LexicalIndex(payloads), Reranker()
    )
    acl = AclPredicate.for_user(_STUB_USERS["cfo-office"])

    print(f"collection      {collection}")
    print(f"chunks          {len(payloads)}")
    print(f"hard questions  {len(hard)}")
    print(f"planted facts   {len(facts)}")

    # ---------------------------------------------------------------- aggregate
    results: dict[str, dict[str, float]] = {}
    latency: dict[str, list[float]] = {}
    stage_ms: dict[str, dict[str, list[float]]] = defaultdict(lambda: defaultdict(list))
    by_hardness: dict[str, dict[str, list[int | None]]] = defaultdict(lambda: defaultdict(list))

    for mode in MODES:
        ranks: list[int | None] = []
        lat: list[float] = []
        for q in hard:
            r = pipeline.run(q["question"], acl, mode=mode, top_k=5, candidates=20)
            rank = rank_of(r.chunks, q["expect_doc_id"], q["expect_version"])
            ranks.append(rank)
            lat.append(r.total_ms)
            by_hardness[q["hardness"]][str(mode)].append(rank)
            for stage, ms in r.stages.items():
                stage_ms[str(mode)][stage].append(ms)
        results[str(mode)] = score(ranks)
        latency[str(mode)] = lat

    print("\n=== HARD SET: retrieval quality by configuration ===")
    print(f"  {'mode':<18}{'R@1':>7}{'R@3':>7}{'R@5':>7}{'MRR':>8}{'p50 ms':>9}{'p95 ms':>9}")
    for mode in MODES:
        m, lat = results[str(mode)], sorted(latency[str(mode)])
        p95 = lat[min(int(len(lat) * 0.95), len(lat) - 1)]
        print(
            f"  {mode!s:<18}{m['recall@1']:>7.3f}{m['recall@3']:>7.3f}"
            f"{m['recall@5']:>7.3f}{m['mrr']:>8.3f}"
            f"{statistics.median(lat):>9.1f}{p95:>9.1f}"
        )

    print("\n=== by hardness class: MRR (why hybrid exists) ===")
    classes = sorted(by_hardness)
    header = "  " + f"{'hardness':<16}" + "".join(f"{str(m)[:13]:>15}" for m in MODES)
    print(header)
    for cls in classes:
        row = f"  {cls:<16}"
        for mode in MODES:
            ranks = by_hardness[cls][str(mode)]
            mrr = sum(1 / r for r in ranks if r) / len(ranks) if ranks else 0.0
            row += f"{mrr:>15.3f}"
        print(row)

    print("\n=== stage latency (ms, mean) ===")
    all_stages = sorted({s for m in stage_ms.values() for s in m})
    print("  " + f"{'mode':<18}" + "".join(f"{s.replace('_ms', ''):>14}" for s in all_stages))
    for mode in MODES:
        row = f"  {mode!s:<18}"
        for stage in all_stages:
            vals = stage_ms[str(mode)].get(stage, [])
            row += f"{statistics.fmean(vals):>14.2f}" if vals else f"{'-':>14}"
        print(row)

    # ------------------------------------------------- regression on the easy set
    print("\n=== PLANTED FACTS (the easy set): did anything regress? ===")
    print(f"  {'mode':<18}{'R@1':>7}{'R@3':>7}{'MRR':>8}")
    for mode in MODES:
        ranks = [
            rank_of(
                pipeline.run(f["question"], acl, mode=mode, top_k=5, candidates=20).chunks,
                f["doc_id"],
                f["doc_version"],
            )
            for f in facts
        ]
        m = score(ranks)
        print(f"  {mode!s:<18}{m['recall@1']:>7.3f}{m['recall@3']:>7.3f}{m['mrr']:>8.3f}")

    # --------------------------------------------------------- ACL still enforced
    print("\n=== ACL holds in every mode (Q-6) ===")
    helios = "What is the indicative offer for Project Helios?"
    print(f"  {'mode':<18}{'contractor':>12}{'analyst':>10}{'cfo':>8}")
    violations = 0
    for mode in MODES:
        row = f"  {mode!s:<18}"
        for user in ("contractor", "analyst-emea", "cfo-office"):
            a = AclPredicate.for_user(_STUB_USERS[user])
            chunks = pipeline.run(helios, a, mode=mode, top_k=5, candidates=20).chunks
            leaked = any(c.classification == "confidential" for c in chunks)
            if leaked and user != "cfo-office":
                violations += 1
            row += (
                f"{'LEAK' if leaked else 'clean':>12}"
                if user == "contractor"
                else (
                    f"{'LEAK' if leaked else 'clean':>10}"
                    if user == "analyst-emea"
                    else f"{'yes' if leaked else 'no':>8}"
                )
            )
        print(row)
    print(f"\n  VIOLATIONS: {violations}   (Q-6 requires 0)")

    return 0 if violations == 0 else 1


if __name__ == "__main__":
    raise SystemExit(main())
