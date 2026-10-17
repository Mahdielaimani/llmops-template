from __future__ import annotations

import json
from typing import Any

import httpx
import pytest

from llmops_eval.generation import (
    AnswerRecord,
    deterministic_summary,
    exact_match,
    judge_answers,
    judge_summary,
)
from llmops_eval.judge import RUBRICS, JudgeClient, extract_json, interpret


def test_extract_json_strips_fences_and_prose() -> None:
    assert extract_json('```json\n{"a": 1}\n```') == {"a": 1}
    assert extract_json('Sure! Here it is: {"a": 1} Hope that helps.') == {"a": 1}


def test_extract_json_without_object_raises() -> None:
    with pytest.raises(ValueError):
        extract_json("the answer is correct")


def test_unknown_label_is_a_schema_error_not_a_guess() -> None:
    status, parsed, error = interpret(
        RUBRICS["correctness"], '{"reason": "x", "verdict": "mostly_right"}'
    )
    assert status == "schema_error" and parsed is None and error


def test_unparseable_output_is_a_parse_error() -> None:
    assert interpret(RUBRICS["relevance"], "relevant")[0] == "parse_error"


def _client(content: str | None = None, *, seen: dict[str, Any] | None = None) -> JudgeClient:
    def handler(req: httpx.Request) -> httpx.Response:
        if seen is not None:
            seen["body"] = json.loads(req.content)
        if content is None:
            raise httpx.ConnectError("refused", request=req)
        return httpx.Response(200, json={"choices": [{"message": {"content": content}}]})

    http = httpx.AsyncClient(base_url="http://judge/v1", transport=httpx.MockTransport(handler))
    return JudgeClient(http, "llama3.2:3b")


async def test_judge_request_is_deterministic_json_mode() -> None:
    seen: dict[str, Any] = {}
    client = _client('{"reason": "on topic", "verdict": "relevant"}', seen=seen)
    res = await client.judge("relevance", question="Q?", answer="A.", context="C", reference="")
    assert res.status == "ok" and res.verdict == "relevant" and res.score == 1.0
    assert res.prompt_ref == "judge/relevance@1.0.0"
    body = seen["body"]
    assert body["temperature"] == 0.0
    assert body["response_format"] == {"type": "json_object"}
    assert body["model"] == "llama3.2:3b"
    # relevance does not declare `context`; it must not leak into the prompt
    assert "## Sources" not in body["messages"][0]["content"]


async def test_faithfulness_score_is_recomputed_from_claims() -> None:
    raw = json.dumps(
        {
            "claims": [
                {"claim": "a", "verdict": "supported", "source_id": "s1"},
                {"claim": "b", "verdict": "unsupported", "source_id": None},
                {"claim": "c", "verdict": "contradicted", "source_id": "s1"},
            ],
            "faithfulness": 0.9,  # the judge's own arithmetic, wrong
        }
    )
    res = await _client(raw).judge("faithfulness", question="q", answer="a", context="c")
    assert res.score == pytest.approx(1 / 3, abs=1e-4)
    assert res.reported_score == 0.9
    assert len(res.claims) == 3


async def test_faithfulness_with_no_claims_is_unscored() -> None:
    res = await _client('{"claims": []}').judge("faithfulness", question="q", answer="", context="")
    assert res.verdict == "no_claims" and res.score is None


async def test_transport_failure_is_recorded() -> None:
    res = await _client(None).judge("relevance", question="q", answer="a")
    assert res.status == "transport_error" and res.error


@pytest.mark.parametrize(
    ("expected", "answer", "abstained", "result"),
    [
        ("EUR 959,065.78", "Revenue was EUR 959,065.78 [x].", False, True),
        ("EUR 959,065.78", "Revenue was EUR959,065.78.", False, True),
        ("EUR 959,065.78", "Revenue was EUR 959,065.79.", False, False),
        ("insufficient_evidence", "insufficient_evidence", True, True),
        ("insufficient_evidence", "EUR 1.", False, False),
        (None, "anything", False, None),
    ],
)
def test_exact_match(expected: str | None, answer: str, abstained: bool, result: bool) -> None:
    assert exact_match(expected, answer, abstained) is result


def _record(**overrides: Any) -> AnswerRecord:
    values: dict[str, Any] = {
        "id": "q1",
        "question": "q",
        "hardness": "planted",
        "expect_doc_id": "D1",
        "expect_answer": "EUR 1",
        "answer": "EUR 1 [D1@v1#c1]",
        "abstained": False,
        "exact_match": True,
        "cited": ["D1@v1#c1"],
        "invalid_citations": [],
        "supplied": ["D1@v1#c1 (p.1)"],
        "context": "ctx",
        "prompt_ref": "rag/rag_answer@1.0.0",
        "generator_model": "qwen2.5:3b",
        "generation_ms": 100.0,
        "input_tokens": 10,
        "output_tokens": 5,
    }
    return AnswerRecord(**{**values, **overrides})


def test_deterministic_summary_separates_abstention_by_answerability() -> None:
    records = [
        _record(),
        _record(id="q2", abstained=True, exact_match=False, answer="insufficient_evidence"),
        _record(
            id="q3",
            expect_doc_id="__none__",
            expect_answer="insufficient_evidence",
            abstained=True,
            exact_match=True,
            cited=[],
        ),
        _record(id="q4", error="ConnectError: refused"),
    ]
    s = deterministic_summary(records)
    assert s["errors"] == 1
    assert s["exact_match"] == pytest.approx(2 / 3, abs=1e-4)
    assert s["abstention_on_answerable"] == 0.5
    assert s["abstention_on_unanswerable"] == 1.0


async def test_correctness_is_skipped_without_reference() -> None:
    client = _client('{"reason": "r", "verdict": "relevant"}')
    rows = [
        {"id": "q1", "hardness": "paraphrase", "question": "q", "answer": "a", "context": "c"},
        {"id": "q2", "error": "boom"},
    ]
    judged = await judge_answers(client, rows, ["correctness", "relevance"])
    assert set(judged[0]["verdicts"]) == {"relevance"}
    assert judged[1]["skipped"] == "generation_error"


def test_judge_summary_reports_failure_rate_beside_mean() -> None:
    def v(status: str, score: float | None) -> dict[str, Any]:
        return {"status": status, "score": score}

    judged = [
        {"verdicts": {"relevance": v("ok", 1.0)}},
        {"verdicts": {"relevance": v("ok", 0.0)}},
        {"verdicts": {"relevance": v("parse_error", None)}},
    ]
    s = judge_summary(judged)["relevance"]
    assert s["mean"] == 0.5 and s["scored"] == 2
    assert s["failure_rate"] == pytest.approx(1 / 3, abs=1e-4)
