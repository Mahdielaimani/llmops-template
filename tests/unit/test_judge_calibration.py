from __future__ import annotations

import json
from collections.abc import Callable
from typing import Any

import httpx
import pytest

from llmops_eval.calibration import (
    agreement_with_exact_match,
    build_probes,
    cohen_kappa,
    position_probe,
    run_probes,
    summarise_probes,
)
from llmops_eval.judge import JudgeClient

PLANTED = {
    "id": "q001",
    "question": "What was Q1 FY2026 revenue for emea sales?",
    "expect_doc_id": "QR-EMEA",
    "expect_answer": "EUR 959,065.78",
    "supplied": ["OTHER@v1#c1 (p.1)", "QR-EMEA@v1#c4 (p.2)"],
    "context": "<source ...>",
}
UNANSWERABLE = {
    "id": "q109",
    "question": "What was Q4 FY2027 revenue for LATAM?",
    "expect_doc_id": "__none__",
    "expect_answer": "insufficient_evidence",
    "supplied": ["OTHER@v1#c1 (p.1)"],
    "context": "<source ...>",
}


def _variants(records: list[dict[str, Any]]) -> dict[str, Any]:
    return {p.variant: p for p in build_probes(records)}


def test_planted_item_yields_five_labelled_probes() -> None:
    v = _variants([PLANTED])
    assert set(v) == {
        "gold",
        "wrong_figure",
        "fabricated_claim",
        "verbose_gold",
        "false_abstention",
    }
    assert v["gold"].answer == "EUR 959,065.78 [QR-EMEA@v1#c4]."
    assert v["wrong_figure"].answer.startswith("EUR 959,065.79")
    # the fabricated claim keeps the key fact, so it separates the two rubrics
    assert v["fabricated_claim"].expect_correct == "correct"
    assert v["fabricated_claim"].expect_faithful is False
    assert v["verbose_gold"].expect_correct == "correct"
    assert v["false_abstention"].expect_faithful is None


def test_wrong_figure_is_a_one_digit_change() -> None:
    gold, wrong = _variants([PLANTED])["gold"].answer, _variants([PLANTED])["wrong_figure"].answer
    assert sum(a != b for a, b in zip(gold, wrong, strict=True)) == 1


def test_unanswerable_item_yields_abstention_and_hallucination() -> None:
    v = _variants([UNANSWERABLE])
    assert set(v) == {"gold_abstention", "hallucinated"}
    assert v["hallucinated"].expect_correct == "incorrect"


def test_item_whose_document_was_not_retrieved_is_skipped() -> None:
    assert build_probes([{**PLANTED, "supplied": ["OTHER@v1#c1 (p.1)"]}]) == []


def test_items_without_reference_are_skipped() -> None:
    assert build_probes([{**PLANTED, "expect_answer": None}]) == []


def _judge(decide: Callable[[str, str], dict[str, Any]]) -> JudgeClient:
    """`decide(prompt_name, rendered_prompt)` returns the JSON the judge emits."""

    def handler(req: httpx.Request) -> httpx.Response:
        content = json.loads(req.content)["messages"][0]["content"]
        name = next(n for n, marker in MARKERS.items() if marker in content)
        body = {"choices": [{"message": {"content": json.dumps(decide(name, content))}}]}
        return httpx.Response(200, json=body)

    http = httpx.AsyncClient(base_url="http://j/v1", transport=httpx.MockTransport(handler))
    return JudgeClient(http, "llama3.2:3b")


MARKERS = {
    "pairwise": "comparing two answers",
    "correctness": "against a reference answer",
    "faithfulness": "supported by its sources",
    "relevance": "addresses the question",
    "completeness": "covers every part",
}


def _always_correct(name: str, _: str) -> dict[str, Any]:
    if name == "faithfulness":
        return {"claims": [{"claim": "x", "verdict": "supported", "source_id": "s"}]}
    if name == "pairwise":
        return {"reason": "r", "verdict": "A"}
    return {"reason": "r", "verdict": "correct"}


async def test_a_judge_that_always_says_correct_catches_no_errors() -> None:
    probes = build_probes([PLANTED, UNANSWERABLE])
    rows = await run_probes(_judge(_always_correct), probes, repeats=1)
    s = summarise_probes(rows)
    assert s["error_recall"] == 0.0
    assert s["false_alarm_rate"] == 0.0
    assert s["variants"]["gold"]["correctness_accuracy"] == 1.0
    assert s["variants"]["wrong_figure"]["correctness_accuracy"] == 0.0
    # fabricated_claim must be caught by faithfulness, and this judge misses it
    assert s["variants"]["fabricated_claim"]["faithfulness_accuracy"] == 0.0


async def test_always_a_is_reported_as_position_bias() -> None:
    bias = await position_probe(_judge(_always_correct), build_probes([PLANTED]))
    assert bias["position"]["position_flip"] == 1
    assert bias["position"]["consistent_correct"] == 0
    assert bias["verbosity"]["prefers_concise"] == 1


async def test_unstable_verdicts_across_repeats_are_counted() -> None:
    calls = {"n": 0}

    def flaky(name: str, content: str) -> dict[str, Any]:
        if name == "correctness":
            calls["n"] += 1
            return {"reason": "r", "verdict": "correct" if calls["n"] % 2 else "incorrect"}
        return _always_correct(name, content)

    probes = [p for p in build_probes([PLANTED]) if p.variant == "gold"]
    rows = await run_probes(_judge(flaky), probes, repeats=2)
    assert summarise_probes(rows)["variants"]["gold"]["unstable_across_repeats"] == 1


def test_agreement_with_exact_match_lists_disagreements() -> None:
    answers = [{"id": "a", "exact_match": True}, {"id": "b", "exact_match": False}]
    judged = [
        {"id": "a", "verdicts": {"correctness": {"status": "ok", "verdict": "correct"}}},
        {"id": "b", "verdicts": {"correctness": {"status": "ok", "verdict": "correct"}}},
    ]
    out = agreement_with_exact_match(answers, judged)
    assert out["agreement"] == 0.5
    assert out["disagreements"] == [{"id": "b", "exact_match": False, "judge": "correct"}]


def test_kappa_is_zero_for_a_constant_rater_on_a_skewed_set() -> None:
    human = ["correct"] * 9 + ["incorrect"]
    judge = ["correct"] * 10
    # 90% raw agreement, no agreement beyond chance
    assert cohen_kappa(human, judge) == 0.0


def test_kappa_perfect_and_undefined() -> None:
    assert cohen_kappa(["a", "b", "a"], ["a", "b", "a"]) == 1.0
    assert cohen_kappa(["a", "a"], ["a", "a"]) is None
    assert cohen_kappa([], []) is None


@pytest.mark.parametrize("bad", [["a"], ["a", "b", "c"]])
def test_kappa_requires_equal_length(bad: list[str]) -> None:
    assert cohen_kappa(["a", "b"], bad) is None
