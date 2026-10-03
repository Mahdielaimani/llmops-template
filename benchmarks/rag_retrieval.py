"""Retrieval quality and latency against the planted facts.

    docker compose --profile core up -d qdrant
    uv run python -m rag_service.corpus && uv run python -m rag_service.ingest
    uv run python benchmarks/rag_retrieval.py

Produces the numbers in docs/rag.md. Scores against ground truth rather than a
judge: every fact was planted at a known (doc_id, version, page), so Recall@k and
MRR are computable without an LLM (docs/evaluation.md, Phase 11, extends this).
"""

from __future__ import annotations

import json
import os
import statistics
import sys
from pathlib import Path

from qdrant_client import QdrantClient

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "apps" / "rag-service" / "src"))

from rag_service.acl import AclPredicate  # noqa: E402
from rag_service.embedding import Embedder  # noqa: E402
from rag_service.main import _STUB_USERS  # noqa: E402
from rag_service.retriever import Retriever, build_context  # noqa: E402

FACTS = ROOT / "data" / "evaluation" / "planted_facts.json"
QDRANT = os.getenv("LLMOPS_QDRANT__URL", "http://localhost:6333")


def percentiles(xs: list[float]) -> dict[str, float]:
    s = sorted(xs)
    return {
        "p50": round(statistics.median(s), 2),
        "p95": round(s[min(int(len(s) * 0.95), len(s) - 1)], 2),
        "mean": round(statistics.fmean(s), 2),
    }


def main() -> int:
    facts = json.loads(FACTS.read_text(encoding="utf-8"))
    client = QdrantClient(url=QDRANT)
    collections = sorted(
        c.name for c in client.get_collections().collections if c.name.startswith("docs_")
    )
    if not collections:
        print("no docs_* collection; run `python -m rag_service.ingest`", file=sys.stderr)
        return 1
    retriever = Retriever(client, Embedder(), collections[-1])

    print(f"collection   {collections[-1]}")
    print(f"facts        {len(facts)}")

    # --- retrieval quality, as the user who can see everything ----------------
    acl_all = AclPredicate.for_user(_STUB_USERS["cfo-office"])
    ranks: list[int | None] = []
    embed_ms, search_ms, total_ms, ctx_chars = [], [], [], []

    for f in facts:
        res = retriever.search(f["question"], acl_all, top_k=10, candidates=20)
        rank = None
        for i, c in enumerate(res.chunks, start=1):
            if c.doc_id == f["doc_id"] and c.doc_version == f["doc_version"]:
                rank = i
                break
        ranks.append(rank)
        embed_ms.append(res.embed_ms)
        search_ms.append(res.search_ms)
        total_ms.append(res.total_ms)
        context, _ = build_context(res.chunks[:5])
        ctx_chars.append(len(context))

    def recall_at(k: int) -> float:
        return sum(1 for r in ranks if r is not None and r <= k) / len(ranks)

    mrr = sum(1 / r for r in ranks if r is not None) / len(ranks)

    print("\n=== retrieval quality (ground truth, no judge) ===")
    for k in (1, 3, 5, 10):
        print(f"  Recall@{k:<3} {recall_at(k):.3f}")
    print(f"  MRR       {mrr:.3f}")
    missed = [f["question"] for f, r in zip(facts, ranks, strict=True) if r is None]
    print(f"  missed    {len(missed)}")
    for q in missed:
        print(f"    - {q}")

    print("\n=== retrieval latency (ms, inside the handler) ===")
    print(f"  embed      {percentiles(embed_ms)}")
    print(f"  search     {percentiles(search_ms)}")
    print(f"  total      {percentiles(total_ms)}")
    print(f"  context chars p50 {percentiles(ctx_chars)['p50']:.0f}")

    # --- ACL: the hard gate ---------------------------------------------------
    print("\n=== ACL enforcement per identity (Q-6: zero unauthorized) ===")
    print(f"  {'user':<17}{'clearance':<14}{'candidates':>11}{'returned':>10}{'leaked':>8}")
    violations = 0
    identities = (
        "contractor",
        "analyst-emea",
        "analyst-apac",
        "auditor",
        "auditor-expired",
        "cfo-office",
    )
    for uid in identities:
        user = _STUB_USERS[uid]
        acl = AclPredicate.for_user(user)
        leaked_total = 0
        cand_total = 0
        ret_total = 0
        for f in facts:
            res = retriever.search(f["question"], acl, top_k=5, candidates=20)
            cand_total += res.candidates_returned
            ret_total += len(res.chunks)
            leaked_total += sum(
                1
                for c in res.chunks
                if not acl.allows(
                    {
                        "classification": c.classification,
                        "classification_level": None,
                        "department": c.department,
                        "doc_id": c.doc_id,
                    }
                )
                or _above_clearance(c.classification, user)
            )
        violations += leaked_total
        print(
            f"  {uid:<17}{user.clearance.name.lower():<14}"
            f"{cand_total:>11}{ret_total:>10}{leaked_total:>8}"
        )
    print(f"\n  TOTAL VIOLATIONS: {violations}   (quality gate Q-6 requires 0)")

    # --- the confidential document, specifically ------------------------------
    print("\n=== Project Helios (confidential) reachability ===")
    q = "What is the indicative offer for Project Helios?"
    for uid in ("contractor", "analyst-emea", "auditor", "cfo-office"):
        acl = AclPredicate.for_user(_STUB_USERS[uid])
        res = retriever.search(q, acl, top_k=5, candidates=20)
        hit = any(c.doc_id == "MA-PROJECT-HELIOS-001" for c in res.chunks)
        print(f"  {uid:<16} candidates={res.candidates_returned:<3} reaches_memo={hit}")

    print("\n=== cache scope hashes (ADR-019: must all differ) ===")
    seen: dict[str, str] = {}
    for uid, user in sorted(_STUB_USERS.items()):
        h = AclPredicate.for_user(user).scope_hash()
        collision = seen.get(h)
        seen[h] = uid
        print(f"  {uid:<17}{h}{'  <-- COLLISION with ' + collision if collision else ''}")

    return 0 if violations == 0 else 1


def _above_clearance(classification: str, user: object) -> bool:
    from rag_service.acl import Classification

    return Classification.parse(classification) > user.clearance


if __name__ == "__main__":
    raise SystemExit(main())
