"""Judge calibration: score the judge against answers whose label is known.

An LLM judge is a measuring instrument, and an uncalibrated instrument reports
numbers, not measurements. The only way to know its error is to hand it items
where the right verdict is known by construction:

    gold              the reference figure, cited          correct,  faithful
    wrong_figure      one digit of the figure changed      incorrect, unfaithful
    fabricated_claim  gold + an invented supporting claim  correct,  unfaithful
    verbose_gold      gold + padding with no new facts     correct,  faithful
    false_abstention  insufficient_evidence, answerable    incorrect
    hallucinated      a figure for an unanswerable item    incorrect, unfaithful
    gold_abstention   insufficient_evidence, unanswerable  correct

`verbose_gold` exists to catch verbosity bias: it must score the same as `gold`.
`fabricated_claim` separates correctness from faithfulness: the key fact is
right, so a judge that marks it incorrect is conflating the two rubrics.

Position bias needs a pairwise question, so the pairwise prompt is asked twice
with the order swapped; a judge without position bias picks the same answer both
times.
"""

from __future__ import annotations

import re
from collections import defaultdict
from dataclasses import dataclass
from typing import Any, Literal

from llmops_eval.judge import JudgeClient
from rag_service.generation import ABSTAIN

_DIGIT = re.compile(r"\d")
_PAGE = re.compile(r"\s*\(p\.\s*\d+\)\s*$")

Label = Literal["correct", "incorrect"]


@dataclass(frozen=True)
class Probe:
    item_id: str
    variant: str
    question: str
    context: str
    reference: str
    answer: str
    expect_correct: Label
    expect_faithful: bool | None  # None: abstentions make no claims to check


def _bump_last_digit(text: str) -> str:
    """Change one digit, so the wrong figure is as close as possible to the right
    one. A judge that only notices gross errors would pass a cruder perturbation."""
    matches = list(_DIGIT.finditer(text))
    if not matches:
        raise ValueError(f"no digit to perturb in {text!r}")
    m = matches[-1]
    replacement = str((int(m.group()) + 1) % 10)
    return text[: m.start()] + replacement + text[m.end() :]


def _citation_for(record: dict[str, Any]) -> str | None:
    for sup in record["supplied"]:
        if sup.startswith(f"{record['expect_doc_id']}@"):
            return _PAGE.sub("", sup)
    return None


def build_probes(records: list[dict[str, Any]]) -> list[Probe]:
    """Probes reuse the generation run's context, so the judge sees exactly what
    a generated answer would have seen. Items whose expected document was not
    retrieved are skipped: a gold answer citing an unseen source is not gold."""
    probes: list[Probe] = []
    for rec in records:
        expected = rec.get("expect_answer")
        if not expected or rec.get("error"):
            continue
        base = {
            "item_id": rec["id"],
            "question": rec["question"],
            "context": rec["context"],
            "reference": expected,
        }
        if expected == ABSTAIN:
            fallback = _PAGE.sub("", rec["supplied"][0]) if rec["supplied"] else "SRC"
            probes += [
                Probe(**base, variant="gold_abstention", answer=ABSTAIN,
                      expect_correct="correct", expect_faithful=None),
                Probe(**base, variant="hallucinated", answer=f"EUR 1,284,330.00 [{fallback}]",
                      expect_correct="incorrect", expect_faithful=False),
            ]  # fmt: skip
            continue
        cite = _citation_for(rec)
        if cite is None:
            continue
        gold = f"{expected} [{cite}]."
        probes += [
            Probe(**base, variant="gold", answer=gold,
                  expect_correct="correct", expect_faithful=True),
            Probe(**base, variant="wrong_figure", answer=f"{_bump_last_digit(expected)} [{cite}].",
                  expect_correct="incorrect", expect_faithful=False),
            Probe(**base, variant="fabricated_claim",
                  answer=f"{gold} This was driven mainly by a 40% rise in new enterprise "
                         f"contracts signed in the period [{cite}].",
                  expect_correct="correct", expect_faithful=False),
            Probe(**base, variant="verbose_gold",
                  answer=f"Based on a careful review of the provided financial sources, and "
                         f"taking into account the reporting period referred to in the "
                         f"question, the figure reported in the sources is {gold} This is "
                         f"the value as stated in the cited source document.",
                  expect_correct="correct", expect_faithful=True),
            Probe(**base, variant="false_abstention", answer=ABSTAIN,
                  expect_correct="incorrect", expect_faithful=None),
        ]  # fmt: skip
    return probes


async def run_probes(judge: JudgeClient, probes: list[Probe], repeats: int) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    for p in probes:
        values = {
            "question": p.question,
            "answer": p.answer,
            "context": p.context,
            "reference": p.reference,
        }
        correctness = [
            (await judge.judge("correctness", **values)).to_dict() for _ in range(repeats)
        ]
        faithfulness = (
            (await judge.judge("faithfulness", **values)).to_dict()
            if p.expect_faithful is not None
            else None
        )
        rows.append(
            {
                "item_id": p.item_id,
                "variant": p.variant,
                "answer": p.answer,
                "expect_correct": p.expect_correct,
                "expect_faithful": p.expect_faithful,
                "correctness": correctness,
                "faithfulness": faithfulness,
            }
        )
    return rows


def _judged_correct(v: dict[str, Any]) -> Label | None:
    if v["status"] != "ok":
        return None
    # partially_correct counts as not correct: the probes are built so that every
    # expected label is unambiguous, and a hedge on an unambiguous item is a miss.
    return "correct" if v["verdict"] == "correct" else "incorrect"


def summarise_probes(rows: list[dict[str, Any]]) -> dict[str, Any]:
    by_variant: dict[str, dict[str, Any]] = defaultdict(
        lambda: {"n": 0, "correctness_hits": 0, "correctness_failed": 0,
                 "faithfulness_hits": 0, "faithfulness_n": 0, "unstable": 0}
    )  # fmt: skip
    tp = fp = tn = fn = 0  # positive = the judge says "incorrect", i.e. an error is flagged
    for r in rows:
        s = by_variant[r["variant"]]
        s["n"] += 1
        labels = [_judged_correct(v) for v in r["correctness"]]
        if len({label for label in labels if label}) > 1:
            s["unstable"] += 1
        first = labels[0]
        if first is None:
            s["correctness_failed"] += 1
        else:
            s["correctness_hits"] += first == r["expect_correct"]
            error_present = r["expect_correct"] == "incorrect"
            flagged = first == "incorrect"
            tp += error_present and flagged
            fn += error_present and not flagged
            fp += not error_present and flagged
            tn += not error_present and not flagged
        f = r["faithfulness"]
        if f and f["status"] == "ok" and f["score"] is not None:
            s["faithfulness_n"] += 1
            judged_faithful = f["score"] >= 1.0
            s["faithfulness_hits"] += judged_faithful == r["expect_faithful"]

    variants = {}
    for name, s in sorted(by_variant.items()):
        scored = s["n"] - s["correctness_failed"]
        variants[name] = {
            "n": s["n"],
            "correctness_accuracy": _rate(s["correctness_hits"], scored),
            "correctness_failed": s["correctness_failed"],
            "faithfulness_accuracy": _rate(s["faithfulness_hits"], s["faithfulness_n"]),
            "unstable_across_repeats": s["unstable"],
        }
    return {
        "variants": variants,
        # Error detection framed as a classifier: how many planted errors the judge
        # catches, and how often it flags a correct answer.
        "error_recall": _rate(tp, tp + fn),
        "false_alarm_rate": _rate(fp, fp + tn),
        "confusion": {"tp": tp, "fn": fn, "fp": fp, "tn": tn},
    }


async def position_probe(judge: JudgeClient, probes: list[Probe]) -> dict[str, Any]:
    """gold vs wrong_figure, asked in both orders. A position-consistent judge picks
    gold both times; picking 'A' both times is pure position preference."""
    pairs: dict[str, dict[str, Probe]] = defaultdict(dict)
    for p in probes:
        if p.variant in ("gold", "wrong_figure", "verbose_gold"):
            pairs[p.item_id][p.variant] = p
    outcome = {"consistent_correct": 0, "consistent_wrong": 0, "position_flip": 0, "failed": 0}
    verbosity = {"prefers_verbose": 0, "prefers_concise": 0, "tie": 0, "failed": 0}
    for item in pairs.values():
        gold, wrong = item.get("gold"), item.get("wrong_figure")
        if gold and wrong:
            first = await _pairwise(judge, gold, gold.answer, wrong.answer)
            second = await _pairwise(judge, gold, wrong.answer, gold.answer)
            if first is None or second is None:
                outcome["failed"] += 1
            elif first == "A" and second == "B":
                outcome["consistent_correct"] += 1
            elif first == "B" and second == "A":
                outcome["consistent_wrong"] += 1
            else:
                outcome["position_flip"] += 1
        verbose = item.get("verbose_gold")
        if gold and verbose:
            pick = await _pairwise(judge, gold, gold.answer, verbose.answer)
            key = {"A": "prefers_concise", "B": "prefers_verbose", "tie": "tie"}
            verbosity[key.get(pick or "", "failed")] += 1
    return {"position": outcome, "verbosity": verbosity}


async def _pairwise(judge: JudgeClient, probe: Probe, answer_a: str, answer_b: str) -> str | None:
    res = await judge.judge(
        "pairwise",
        question=probe.question,
        context=probe.context,
        reference=probe.reference,
        answer_a=answer_a,
        answer_b=answer_b,
    )
    return res.verdict if res.status == "ok" else None


def agreement_with_exact_match(
    answers: list[dict[str, Any]], judged: list[dict[str, Any]]
) -> dict[str, Any]:
    """Judge correctness against the deterministic check on real generated answers.
    Exact match is not perfect either (it misses a correctly reformatted figure),
    so disagreements are listed for a human to read, not counted as judge errors."""
    em = {a["id"]: a.get("exact_match") for a in answers}
    agree, disagree = 0, []
    for row in judged:
        v = row.get("verdicts", {}).get("correctness")
        truth = em.get(row["id"])
        if not v or truth is None or v["status"] != "ok":
            continue
        if (v["verdict"] == "correct") == truth:
            agree += 1
        else:
            disagree.append({"id": row["id"], "exact_match": truth, "judge": v["verdict"]})
    n = agree + len(disagree)
    return {"n": n, "agreement": _rate(agree, n), "disagreements": disagree}


def cohen_kappa(a: list[str], b: list[str]) -> float | None:
    """Agreement beyond chance. Raw agreement flatters a judge on a skewed set: if
    90% of answers are correct, a judge that always says 'correct' agrees 90%."""
    if len(a) != len(b) or not a:
        return None
    n = len(a)
    observed = sum(x == y for x, y in zip(a, b, strict=True)) / n
    labels = set(a) | set(b)
    expected = sum((a.count(lab) / n) * (b.count(lab) / n) for lab in labels)
    if expected == 1.0:
        return None  # both raters used one identical label; kappa is undefined
    return round((observed - expected) / (1 - expected), 4)


def _rate(num: int, den: int) -> float | None:
    return round(num / den, 4) if den else None
