"""Run the dataset through a retrieval configuration and score it.

A run records the configuration it measured, not just the numbers: dataset
version and hash, index version, embedding model, reranker, retrieval mode,
top_k. A metric without that attribution cannot be compared to anything
(ADR-015 class 8), which is the mistake Phase 9's 12-question table was one step
away from making.
"""

from __future__ import annotations

import json
import time
from dataclasses import asdict, dataclass, field
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

from qdrant_client import QdrantClient

from llmops_core.logging import get_logger
from llmops_eval.metrics import Judgement, MetricSet, by_hardness, check_gates, summarise
from rag_service.acl import AclPredicate, Classification
from rag_service.embedding import Embedder
from rag_service.fusion import Reranker
from rag_service.lexical import LexicalIndex, load_payloads
from rag_service.main import _STUB_USERS
from rag_service.pipeline import Mode, RetrievalPipeline
from rag_service.retriever import Retriever

log = get_logger("eval")

RESULTS_DIR = Path(__file__).resolve().parents[4] / "evaluations"
DATASET = Path(__file__).resolve().parents[4] / "data" / "evaluation" / "dataset.json"

# The identity used for quality measurement: sees everything, so a miss is a
# retrieval failure rather than an authorization decision. ACL is measured
# separately, against every identity.
QUALITY_IDENTITY = "cfo-office"


@dataclass
class RunConfig:
    mode: str
    top_k: int = 5
    candidates: int = 20
    dataset_version: str = ""
    dataset_sha: str = ""
    index_version: str = ""
    embed_model: str = ""
    reranker: str = ""
    identity: str = QUALITY_IDENTITY


@dataclass
class RunResult:
    run_id: str
    started_at: str
    config: RunConfig
    metrics: dict[str, Any]
    per_hardness: dict[str, dict[str, Any]]
    gate_failures: list[str]
    latency_ms: dict[str, float]
    duration_s: float
    acl_by_identity: dict[str, int] = field(default_factory=dict)

    @property
    def passed(self) -> bool:
        return not self.gate_failures


def load_dataset(path: Path = DATASET) -> tuple[list[dict[str, Any]], str, str]:
    raw = json.loads(path.read_text(encoding="utf-8"))
    return raw["items"], str(raw["version"]), str(raw["sha256"])


def build_pipeline(qdrant_url: str) -> tuple[RetrievalPipeline, str, str, str]:
    client = QdrantClient(url=qdrant_url)
    collections = sorted(
        c.name for c in client.get_collections().collections if c.name.startswith("docs_")
    )
    if not collections:
        raise RuntimeError(f"no docs_* collection in {qdrant_url}; run rag_service.ingest")
    collection = collections[-1]
    embedder = Embedder()
    reranker = Reranker()
    pipeline = RetrievalPipeline(
        Retriever(client, embedder, collection),
        LexicalIndex(load_payloads(client, collection)),
        reranker,
    )
    return pipeline, collection.removeprefix("docs_"), embedder.model_name, reranker.model_name


def _percentiles(values: list[float]) -> dict[str, float]:
    if not values:
        return {}
    s = sorted(values)
    return {
        "p50": round(s[len(s) // 2], 2),
        "p95": round(s[min(int(len(s) * 0.95), len(s) - 1)], 2),
        "mean": round(sum(s) / len(s), 2),
    }


def run(
    pipeline: RetrievalPipeline,
    items: list[dict[str, Any]],
    config: RunConfig,
    *,
    check_acl: bool = True,
) -> RunResult:
    started = time.perf_counter()
    acl = AclPredicate.for_user(_STUB_USERS[config.identity])
    judgements: list[Judgement] = []
    latencies: list[float] = []

    for item in items:
        unanswerable = item["expect_doc_id"] == "__none__"
        result = pipeline.run(
            item["question"],
            acl,
            mode=Mode(config.mode),
            top_k=config.top_k,
            candidates=config.candidates,
        )
        latencies.append(result.total_ms)

        rank: int | None = None
        relevant = 0
        for i, chunk in enumerate(result.chunks, start=1):
            hit = chunk.doc_id == item["expect_doc_id"] and (
                item["expect_version"] is None or chunk.doc_version == item["expect_version"]
            )
            if hit:
                relevant += 1
                rank = rank or i

        judgements.append(
            Judgement(
                item_id=item["id"],
                hardness=item["hardness"],
                rank=None if unanswerable else rank,
                retrieved=len(result.chunks),
                relevant_retrieved=relevant,
                expected_unanswerable=unanswerable,
                # Retrieval cannot abstain; abstention is a generation behaviour and
                # is measured from Phase 15. Recorded as False rather than guessed.
                abstained=False,
            )
        )

    # ACL is measured across every identity, not only the quality one: the gate is
    # "no identity ever retrieves above its clearance", which one identity cannot show.
    acl_by_identity: dict[str, int] = {}
    if check_acl:
        for name, user in sorted(_STUB_USERS.items()):
            predicate = AclPredicate.for_user(user)
            violations = 0
            for item in items:
                res = pipeline.run(
                    item["question"],
                    predicate,
                    mode=Mode(config.mode),
                    top_k=config.top_k,
                    candidates=config.candidates,
                )
                for chunk in res.chunks:
                    level = Classification.parse(chunk.classification)
                    if (
                        predicate.expired
                        or level > predicate.clearance
                        or (
                            level is not Classification.PUBLIC
                            and chunk.department not in predicate.departments
                        )
                    ):
                        violations += 1
            acl_by_identity[name] = violations
        total = sum(acl_by_identity.values())
        if total:
            judgements[0].acl_violations = total

    metrics = summarise(judgements, top_k=config.top_k)
    run_result = RunResult(
        run_id=f"{config.mode}-{datetime.now(UTC).strftime('%Y%m%dT%H%M%S')}",
        started_at=datetime.now(UTC).isoformat(timespec="seconds"),
        config=config,
        metrics=metrics.as_dict(),
        per_hardness={k: v.as_dict() for k, v in by_hardness(judgements).items()},
        gate_failures=check_gates(metrics),
        latency_ms=_percentiles(latencies),
        duration_s=round(time.perf_counter() - started, 2),
        acl_by_identity=acl_by_identity,
    )
    log.info(
        "eval_run",
        run_id=run_result.run_id,
        mode=config.mode,
        n=metrics.n,
        recall_at_5=metrics.recall_at_5,
        mrr=metrics.mrr,
        acl_violations=metrics.acl_violations,
        passed=run_result.passed,
    )
    return run_result


def save(result: RunResult, directory: Path = RESULTS_DIR) -> Path:
    directory.mkdir(parents=True, exist_ok=True)
    path = directory / f"{result.run_id}.json"
    path.write_text(json.dumps(asdict(result), indent=2), encoding="utf-8")
    return path


def baseline_path(directory: Path = RESULTS_DIR) -> Path:
    """The run future changes are compared against (Phase 14). Kept as a separate
    file rather than "the most recent run" so a regression cannot silently become
    the new baseline."""
    return directory / "baseline.json"


def load_baseline(directory: Path = RESULTS_DIR) -> RunResult | None:
    path = baseline_path(directory)
    if not path.exists():
        return None
    raw = json.loads(path.read_text(encoding="utf-8"))
    raw["config"] = RunConfig(**raw["config"])
    return RunResult(**raw)


def compare(current: MetricSet | dict[str, Any], baseline: dict[str, Any]) -> list[str]:
    """Regression lines for Phase 14. Reports improvements too, because a change
    that improves one metric and damages another is the case the gate must catch
    (CLAUDE.md §37)."""
    cur = current.as_dict() if isinstance(current, MetricSet) else current
    lines: list[str] = []
    for key in ("recall_at_1", "recall_at_5", "mrr", "ndcg_at_5", "context_precision"):
        before, after = float(baseline.get(key, 0)), float(cur.get(key, 0))
        delta = after - before
        if abs(delta) >= 0.005:
            lines.append(f"{key}: {before:.4f} -> {after:.4f} ({delta:+.4f})")
    before_acl = int(baseline.get("acl_violations", 0))
    after_acl = int(cur.get("acl_violations", 0))
    if after_acl != before_acl:
        lines.append(f"acl_violations: {before_acl} -> {after_acl}")
    return lines
