"""Evaluation metrics and gates.

Metrics are arithmetic, so they get arithmetic tests — a metric implementation
that is subtly wrong produces numbers nobody can check, and every conclusion
drawn from them inherits the error. The Q-6 gate tests matter most: that gate is
the one allowed to fail a run on its own.
"""

from __future__ import annotations

import json
from pathlib import Path

import pytest

from llmops_eval.metrics import (
    GATES,
    Judgement,
    by_hardness,
    check_gates,
    context_precision,
    context_recall,
    mrr,
    ndcg_at,
    recall_at,
    summarise,
)
from llmops_eval.runner import DATASET, compare, load_dataset


def j(
    rank: int | None,
    *,
    hardness: str = "planted",
    retrieved: int = 5,
    relevant: int = 1,
    unanswerable: bool = False,
    acl: int = 0,
    item_id: str = "q1",
) -> Judgement:
    return Judgement(
        item_id=item_id,
        hardness=hardness,
        rank=rank,
        retrieved=retrieved,
        relevant_retrieved=relevant,
        expected_unanswerable=unanswerable,
        acl_violations=acl,
    )


# ------------------------------------------------------------------- arithmetic


def test_recall_at_counts_hits_within_k() -> None:
    js = [j(1), j(3), j(7), j(None)]
    assert recall_at(js, 1) == 0.25
    assert recall_at(js, 3) == 0.50
    assert recall_at(js, 10) == 0.75


def test_mrr_is_the_mean_reciprocal_rank() -> None:
    assert mrr([j(1), j(2), j(4)]) == pytest.approx((1 + 0.5 + 0.25) / 3)


def test_a_miss_contributes_zero_not_a_skip() -> None:
    """A missed question must drag the mean down. Dropping misses from the
    denominator would make a retriever that finds nothing look perfect."""
    assert mrr([j(1), j(None)]) == pytest.approx(0.5)
    assert recall_at([j(1), j(None)], 5) == 0.5


def test_ndcg_discounts_lower_ranks_more_gently_than_mrr() -> None:
    """Why both are reported: at rank 3 MRR gives 0.333 and nDCG 0.5, and with a
    five-chunk context window rank 3 is not a third as good as rank 1."""
    assert mrr([j(3)]) == pytest.approx(1 / 3)
    assert ndcg_at([j(3)], 5) == pytest.approx(0.5)
    assert ndcg_at([j(1)], 5) == 1.0


def test_ndcg_ignores_hits_beyond_k() -> None:
    assert ndcg_at([j(9)], 5) == 0.0


def test_unanswerable_items_are_excluded_from_retrieval_metrics() -> None:
    """There is nothing to recall. Including them would silently reward or punish
    abstention inside a retrieval metric."""
    js = [j(1), j(None, unanswerable=True)]
    assert recall_at(js, 5) == 1.0
    assert mrr(js) == 1.0
    assert summarise(js).n_answerable == 1


def test_context_precision_measures_prompt_waste() -> None:
    """One relevant chunk out of five is 0.2 — paid for in prefill tokens and in
    the chance the model answers from the wrong document."""
    assert context_precision([j(1, retrieved=5, relevant=1)]) == pytest.approx(0.2)
    assert context_precision([j(1, retrieved=5, relevant=5)]) == 1.0


def test_context_recall_is_whether_the_evidence_made_it_in() -> None:
    assert context_recall([j(1, relevant=1), j(None, relevant=0)]) == 0.5


def test_summarise_records_which_items_were_missed() -> None:
    """A score tells you how bad; the ids tell you what to look at."""
    m = summarise([j(1, item_id="a"), j(None, item_id="b"), j(None, item_id="c")])
    assert m.missed == ["b", "c"]
    assert m.as_dict()["missed_count"] == 2


def test_by_hardness_splits_without_losing_items() -> None:
    js = [j(1, hardness="lexical"), j(2, hardness="paraphrase"), j(None, hardness="paraphrase")]
    per = by_hardness(js)
    assert per["lexical"].n == 1
    assert per["paraphrase"].n == 2
    assert sum(m.n for m in per.values()) == len(js)


# ------------------------------------------------------------------------ gates


def test_acl_violation_fails_the_run_on_its_own() -> None:
    """Q-6 is a hard gate: a run with perfect retrieval and one unauthorized
    retrieval still fails (docs/business-requirements.md §4)."""
    perfect_but_leaking = summarise([j(1, acl=1)])
    failures = check_gates(perfect_but_leaking)
    assert any("Q-6" in f for f in failures)
    assert perfect_but_leaking.recall_at_5 == 1.0  # quality was perfect


def test_gates_pass_on_a_clean_run() -> None:
    assert check_gates(summarise([j(1), j(1), j(2)])) == []


def test_recall_below_threshold_fails_q1() -> None:
    metrics = summarise([j(1)] + [j(None)] * 4)  # recall@5 = 0.2
    failures = check_gates(metrics)
    assert any("Q-1" in f for f in failures)


def test_gate_thresholds_match_the_requirements_doc() -> None:
    """Pinned so a threshold cannot be quietly relaxed to make a run pass."""
    assert GATES["Q-1"] == ("recall_at_5", 0.80)
    assert GATES["Q-2"] == ("mrr", 0.65)


# ------------------------------------------------------------------- comparison


def test_compare_reports_both_regressions_and_improvements() -> None:
    """CLAUDE.md §37: the case to catch is a change that improves one metric and
    damages another, so improvements cannot be filtered out."""
    baseline = {"recall_at_1": 0.90, "mrr": 0.90, "acl_violations": 0}
    current = {"recall_at_1": 0.95, "mrr": 0.70, "acl_violations": 0}
    lines = compare(current, baseline)
    assert any("recall_at_1" in line and "+0.0500" in line for line in lines)
    assert any("mrr" in line and "-0.2000" in line for line in lines)


def test_compare_ignores_noise_below_half_a_point() -> None:
    baseline = {"recall_at_1": 0.9000, "mrr": 0.9000}
    assert compare({"recall_at_1": 0.9020, "mrr": 0.8990}, baseline) == []


def test_compare_always_reports_an_acl_change() -> None:
    assert compare({"acl_violations": 1}, {"acl_violations": 0}) == ["acl_violations: 0 -> 1"]


# --------------------------------------------------------------- the real dataset


def test_the_committed_dataset_is_large_enough_to_resolve_a_difference() -> None:
    """Phase 9 compared five configurations on 12 questions, where one question was
    0.083 MRR and differences under ~0.15 were noise. That is why this exists."""
    items, _version, _sha = load_dataset()
    assert len(items) >= 100
    assert 1 / len(items) < 0.01


def test_the_dataset_hash_matches_its_contents() -> None:
    """A metric is meaningless without the dataset version it was measured on
    (ADR-015 class 8), so the hash has to be checkable."""
    from rag_service.evalset import build_dataset, dataset_sha

    raw = json.loads(Path(DATASET).read_text(encoding="utf-8"))
    assert raw["sha256"] == dataset_sha(build_dataset())
    assert raw["n"] == len(raw["items"])


def test_every_item_has_scorable_ground_truth() -> None:
    items, _, _ = load_dataset()
    for item in items:
        assert item["id"] and item["question"].strip()
        assert item["expect_doc_id"], item["id"]
        assert item["hardness"], item["id"]
        if item["expect_doc_id"] == "__none__":
            assert item["expect_answer"] == "insufficient_evidence"


def test_the_dataset_covers_the_discriminating_classes() -> None:
    """Phase 9 showed `lexical` is easy for BM25 and `near-duplicate` easy for
    everything. Resolution comes from paraphrase and version, so those cannot be
    token categories."""
    items, _, _ = load_dataset()
    counts: dict[str, int] = {}
    for item in items:
        counts[item["hardness"]] = counts.get(item["hardness"], 0) + 1
    assert counts.get("paraphrase", 0) >= 20
    assert counts.get("version", 0) >= 5
    assert counts.get("unanswerable", 0) >= 5


def test_the_dataset_spans_all_four_classifications() -> None:
    """Otherwise the same run cannot measure ACL enforcement."""
    items, _, _ = load_dataset()
    assert {i["min_clearance"] for i in items} == {
        "public",
        "internal",
        "restricted",
        "confidential",
    }


def test_the_committed_baseline_passed_its_gates() -> None:
    """`baseline` refuses to record a failing run, so this asserts the invariant
    rather than the numbers — a regression cannot become the new baseline."""
    from llmops_eval.runner import load_baseline

    baseline = load_baseline()
    if baseline is None:
        pytest.skip("no baseline recorded yet")
    assert baseline.gate_failures == []
    assert baseline.metrics["acl_violations"] == 0
    assert float(baseline.metrics["recall_at_5"]) >= 0.80
