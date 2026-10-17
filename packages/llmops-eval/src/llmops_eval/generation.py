"""Generation evaluation: produce answers once, score them twice.

    generate  dataset -> retrieval -> rag_answer -> llm-application   -> answers.jsonl
    judge     answers.jsonl -> judge model (different family)          -> judged.jsonl

The split is the point. Answers are an artifact: judging the same file with a
second judge, a new judge prompt, or a human gives comparable numbers, because
the thing being measured did not change between measurements. Regenerating on
every judge run would mix generator variance into judge variance.

Deterministic metrics are computed at generation time and never delegated to the
judge: exact figure match against planted references, abstention on unanswerable
items, citation validity. Those are what the judge is later calibrated against.
"""

from __future__ import annotations

import json
import re
from collections import Counter, defaultdict
from dataclasses import asdict, dataclass
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

from llmops_eval.judge import RUBRICS, JudgeClient
from llmops_eval.runner import QUALITY_IDENTITY, RESULTS_DIR
from rag_service.acl import AclPredicate
from rag_service.generation import ABSTAIN, Generator
from rag_service.main import _STUB_USERS
from rag_service.pipeline import Mode, RetrievalPipeline
from rag_service.retriever import build_context

ANSWERS_DIR = RESULTS_DIR / "answers"
JUDGED_DIR = RESULTS_DIR / "judged"
CALIBRATION_DIR = RESULTS_DIR / "calibration"
_SPACE = re.compile(r"\s+")


def _norm(text: str) -> str:
    return _SPACE.sub("", text).lower()


def exact_match(expected: str | None, answer: str, abstained: bool) -> bool | None:
    """None when the item has no reference: most of the dataset was written to test
    retrieval and carries an expected document, not an expected answer. Reporting
    those as failures would invent a correctness number."""
    if not expected:
        return None
    if expected == ABSTAIN:
        return abstained
    return _norm(expected) in _norm(answer)


@dataclass
class AnswerRecord:
    id: str
    question: str
    hardness: str
    expect_doc_id: str
    expect_answer: str | None
    answer: str
    abstained: bool
    exact_match: bool | None
    cited: list[str]
    invalid_citations: list[str]
    supplied: list[str]
    context: str
    prompt_ref: str
    generator_model: str
    generation_ms: float
    input_tokens: int | None
    output_tokens: int | None
    error: str | None = None


async def generate(
    pipeline: RetrievalPipeline,
    generator: Generator,
    items: list[dict[str, Any]],
    *,
    mode: str,
    top_k: int,
    candidates: int,
) -> list[AnswerRecord]:
    acl = AclPredicate.for_user(_STUB_USERS[QUALITY_IDENTITY])
    records: list[AnswerRecord] = []
    for item in items:
        result = pipeline.run(
            item["question"], acl, mode=Mode(mode), top_k=top_k, candidates=candidates
        )
        context, supplied = build_context(result.chunks)
        base: dict[str, Any] = {
            "id": item["id"],
            "question": item["question"],
            "hardness": item["hardness"],
            "expect_doc_id": item["expect_doc_id"],
            "expect_answer": item.get("expect_answer"),
            "supplied": supplied,
            "context": context,
            "prompt_ref": generator.prompt.ref,
        }
        try:
            gen = await generator.generate(item["question"], context, supplied)
        except Exception as exc:  # recorded per item; one failure must not end the run
            records.append(
                AnswerRecord(
                    **base,
                    answer="",
                    abstained=False,
                    exact_match=None,
                    cited=[],
                    invalid_citations=[],
                    generator_model="",
                    generation_ms=0.0,
                    input_tokens=None,
                    output_tokens=None,
                    error=f"{type(exc).__name__}: {exc}",
                )
            )
            continue
        records.append(
            AnswerRecord(
                **base,
                answer=gen.answer,
                abstained=gen.abstained,
                exact_match=exact_match(item.get("expect_answer"), gen.answer, gen.abstained),
                cited=gen.citations.cited,
                invalid_citations=gen.citations.invalid,
                generator_model=gen.model,
                generation_ms=gen.latency_ms,
                input_tokens=gen.input_tokens,
                output_tokens=gen.output_tokens,
            )
        )
    return records


def deterministic_summary(records: list[AnswerRecord]) -> dict[str, Any]:
    ok = [r for r in records if r.error is None]
    referenced = [r for r in ok if r.exact_match is not None]
    answered = [r for r in ok if not r.abstained]
    unanswerable = [r for r in ok if r.expect_answer == ABSTAIN]
    answerable = [r for r in ok if r.expect_doc_id != "__none__"]
    return {
        "n": len(records),
        "errors": len(records) - len(ok),
        "n_with_reference": len(referenced),
        "exact_match": _rate(sum(bool(r.exact_match) for r in referenced), len(referenced)),
        # Abstaining on an answerable item is a recall failure the user experiences
        # as "the assistant doesn't know", even when retrieval found the document.
        "abstention_on_answerable": _rate(sum(r.abstained for r in answerable), len(answerable)),
        "abstention_on_unanswerable": _rate(
            sum(r.abstained for r in unanswerable), len(unanswerable)
        ),
        "answers_with_citation": _rate(sum(bool(r.cited) for r in answered), len(answered)),
        "answers_with_invalid_citation": _rate(
            sum(bool(r.invalid_citations) for r in answered), len(answered)
        ),
        "generation_ms_p50": _p50([r.generation_ms for r in ok]),
        "output_tokens_mean": _mean([r.output_tokens for r in ok if r.output_tokens]),
    }


async def judge_answers(
    judge: JudgeClient, records: list[dict[str, Any]], rubrics: list[str]
) -> list[dict[str, Any]]:
    judged: list[dict[str, Any]] = []
    for rec in records:
        verdicts: dict[str, Any] = {}
        if rec.get("error"):
            judged.append({"id": rec["id"], "skipped": "generation_error", "verdicts": {}})
            continue
        for name in rubrics:
            if RUBRICS[name].needs_reference and not rec.get("expect_answer"):
                continue
            res = await judge.judge(
                name,
                question=rec["question"],
                answer=rec["answer"],
                context=rec["context"],
                reference=rec.get("expect_answer") or "",
            )
            verdicts[name] = res.to_dict()
        judged.append({"id": rec["id"], "hardness": rec["hardness"], "verdicts": verdicts})
    return judged


def judge_summary(judged: list[dict[str, Any]]) -> dict[str, Any]:
    scores: dict[str, list[float]] = defaultdict(list)
    status: dict[str, Counter[str]] = defaultdict(Counter)
    for row in judged:
        for name, v in row["verdicts"].items():
            status[name][v["status"]] += 1
            if v["status"] == "ok" and v["score"] is not None:
                scores[name].append(v["score"])
    out: dict[str, Any] = {}
    for name in sorted(status):
        total = sum(status[name].values())
        out[name] = {
            "n": total,
            "scored": len(scores[name]),
            "mean": _mean(scores[name]),
            # The failure rate is reported next to the mean because the mean is
            # only over what parsed; a judge that fails on hard items inflates it.
            "failure_rate": _rate(total - status[name]["ok"], total),
            "status": dict(status[name]),
        }
    return out


def new_run_id(prefix: str) -> str:
    return f"{prefix}-{datetime.now(UTC).strftime('%Y%m%dT%H%M%S')}"


def save_jsonl(path: Path, header: dict[str, Any], rows: list[dict[str, Any]]) -> Path:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as f:
        f.write(json.dumps({"_header": header}, ensure_ascii=False) + "\n")
        for row in rows:
            f.write(json.dumps(row, ensure_ascii=False) + "\n")
    return path


def load_jsonl(path: Path) -> tuple[dict[str, Any], list[dict[str, Any]]]:
    lines = [json.loads(line) for line in path.read_text(encoding="utf-8").splitlines() if line]
    header = lines[0].get("_header", {}) if lines else {}
    return header, [row for row in lines if "_header" not in row]


def records_to_rows(records: list[AnswerRecord]) -> list[dict[str, Any]]:
    return [asdict(r) for r in records]


def _rate(num: int, den: int) -> float | None:
    return round(num / den, 4) if den else None


def _mean(values: list[float] | list[int]) -> float | None:
    return round(sum(values) / len(values), 4) if values else None


def _p50(values: list[float]) -> float | None:
    return round(sorted(values)[len(values) // 2], 1) if values else None
