"""Retrieval metrics, computed against ground truth rather than judged.

Every planted fact has a known `(doc_id, version)`, so Recall@k, MRR and nDCG are
arithmetic — no model opinion involved. That matters because an LLM judge
(Phase 12) has bias and cost, and anything computable without one should be.

Context precision and context recall are included because they answer the
question Recall@k does not: *of what we put in the prompt, how much was useful,
and did we fit everything useful in?* Those drive token cost and the
`insufficient_evidence` rate respectively.
"""

from __future__ import annotations

import math
from dataclasses import dataclass, field
from typing import Any


@dataclass
class Judgement:
    """One question's outcome. `rank` is None when the expected document never
    appeared — a miss, not a low score."""

    item_id: str
    hardness: str
    rank: int | None
    retrieved: int
    relevant_retrieved: int
    expected_unanswerable: bool = False
    abstained: bool = False
    acl_violations: int = 0


def recall_at(judgements: list[Judgement], k: int) -> float:
    """Fraction of answerable questions whose expected document is in the top k.

    Unanswerable items are excluded: there is nothing to recall, and including
    them would silently reward or punish abstention inside a retrieval metric.
    """
    answerable = [j for j in judgements if not j.expected_unanswerable]
    if not answerable:
        return 0.0
    return sum(1 for j in answerable if j.rank is not None and j.rank <= k) / len(answerable)


def mrr(judgements: list[Judgement]) -> float:
    answerable = [j for j in judgements if not j.expected_unanswerable]
    if not answerable:
        return 0.0
    return sum(1 / j.rank for j in answerable if j.rank) / len(answerable)


def ndcg_at(judgements: list[Judgement], k: int) -> float:
    """Single-relevant-document nDCG, so ideal DCG is 1 and this reduces to
    `1 / log2(rank + 1)`. Reported alongside MRR because it discounts rank 2-5
    more gently, which matters when the context window holds five chunks anyway.
    """
    answerable = [j for j in judgements if not j.expected_unanswerable]
    if not answerable:
        return 0.0
    total = 0.0
    for j in answerable:
        if j.rank is not None and j.rank <= k:
            total += 1.0 / math.log2(j.rank + 1)
    return total / len(answerable)


def precision_at(judgements: list[Judgement], k: int) -> float:
    """With one relevant document per question this is bounded by 1/k, so it is
    reported for completeness rather than as a target. Context precision below is
    the metric that actually says something about prompt waste."""
    answerable = [j for j in judgements if not j.expected_unanswerable]
    if not answerable:
        return 0.0
    return sum(1 / k for j in answerable if j.rank is not None and j.rank <= k) / len(answerable)


def context_precision(judgements: list[Judgement]) -> float:
    """Of the chunks placed in the prompt, the fraction that were relevant.

    Low precision is paid for twice: in tokens (prefill cost, metrics-and-capacity
    §2) and in distraction — an irrelevant chunk is an opportunity for the model
    to answer from the wrong document.
    """
    answerable = [j for j in judgements if not j.expected_unanswerable and j.retrieved]
    if not answerable:
        return 0.0
    return sum(j.relevant_retrieved / j.retrieved for j in answerable) / len(answerable)


def context_recall(judgements: list[Judgement]) -> float:
    """Did the relevant document make it into the prompt at all. With one
    relevant document per question this equals Recall@top_k, and it is named
    separately because that equality stops holding the moment a question needs
    two documents."""
    answerable = [j for j in judgements if not j.expected_unanswerable]
    if not answerable:
        return 0.0
    return sum(1 for j in answerable if j.relevant_retrieved > 0) / len(answerable)


def abstention_rate(judgements: list[Judgement]) -> float:
    unanswerable = [j for j in judgements if j.expected_unanswerable]
    if not unanswerable:
        return 0.0
    return sum(1 for j in unanswerable if j.abstained) / len(unanswerable)


def false_abstention_rate(judgements: list[Judgement]) -> float:
    """Declining a question that *was* answerable. The failure mode that makes an
    abstaining system useless, and the reason coverage is reported against
    accuracy rather than instead of it."""
    answerable = [j for j in judgements if not j.expected_unanswerable]
    if not answerable:
        return 0.0
    return sum(1 for j in answerable if j.abstained) / len(answerable)


@dataclass
class MetricSet:
    n: int
    n_answerable: int
    n_unanswerable: int
    recall_at_1: float
    recall_at_3: float
    recall_at_5: float
    recall_at_10: float
    mrr: float
    ndcg_at_5: float
    precision_at_5: float
    context_precision: float
    context_recall: float
    abstention_rate: float
    false_abstention_rate: float
    acl_violations: int
    missed: list[str] = field(default_factory=list)

    def as_dict(self) -> dict[str, Any]:
        d: dict[str, Any] = {k: v for k, v in self.__dict__.items() if k != "missed"}
        d["missed_count"] = len(self.missed)
        return d


def summarise(judgements: list[Judgement], *, top_k: int = 5) -> MetricSet:
    answerable = [j for j in judgements if not j.expected_unanswerable]
    return MetricSet(
        n=len(judgements),
        n_answerable=len(answerable),
        n_unanswerable=len(judgements) - len(answerable),
        recall_at_1=round(recall_at(judgements, 1), 4),
        recall_at_3=round(recall_at(judgements, 3), 4),
        recall_at_5=round(recall_at(judgements, 5), 4),
        recall_at_10=round(recall_at(judgements, 10), 4),
        mrr=round(mrr(judgements), 4),
        ndcg_at_5=round(ndcg_at(judgements, 5), 4),
        precision_at_5=round(precision_at(judgements, top_k), 4),
        context_precision=round(context_precision(judgements), 4),
        context_recall=round(context_recall(judgements), 4),
        abstention_rate=round(abstention_rate(judgements), 4),
        false_abstention_rate=round(false_abstention_rate(judgements), 4),
        acl_violations=sum(j.acl_violations for j in judgements),
        missed=[j.item_id for j in answerable if j.rank is None],
    )


def by_hardness(judgements: list[Judgement]) -> dict[str, MetricSet]:
    """Per-class breakdown. Phase 9's whole lesson: the aggregate hid the
    mechanism, and the per-class table showed BM25 scoring zero on paraphrase."""
    classes: dict[str, list[Judgement]] = {}
    for j in judgements:
        classes.setdefault(j.hardness, []).append(j)
    return {name: summarise(items) for name, items in sorted(classes.items())}


# Thresholds from docs/business-requirements.md §4. Q-6 is a hard gate: any
# unauthorized retrieval fails the run regardless of every other number.
GATES: dict[str, tuple[str, float]] = {
    "Q-1": ("recall_at_5", 0.80),
    "Q-2": ("mrr", 0.65),
}
HARD_GATE_ACL = "Q-6"


def check_gates(metrics: MetricSet) -> list[str]:
    failures: list[str] = []
    if metrics.acl_violations > 0:
        failures.append(
            f"{HARD_GATE_ACL}: {metrics.acl_violations} unauthorized retrieval(s); "
            f"the threshold is 0 and this fails the run on its own"
        )
    for gate, (field_name, threshold) in GATES.items():
        actual = float(getattr(metrics, field_name))
        if actual < threshold:
            failures.append(f"{gate}: {field_name} {actual:.4f} < {threshold}")
    return failures
