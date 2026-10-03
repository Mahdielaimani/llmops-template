"""Synthetic financial corpus with deliberately planted facts.

    uv run python -m rag_service.corpus

The lab has no real financial documents (docs/business-requirements.md §6), and
synthetic data here is a feature rather than a compromise: facts are planted at
known locations, so retrieval can be scored against ground truth instead of
judged. Each planted fact becomes an evaluation item in Phase 11.

Three properties the corpus must have, because the platform claims to handle them:
  - **versions** — the same document restated, so "what did Q2 say before the
    restatement" has a correct answer
  - **classifications** — public through confidential, so the ACL has something
    to enforce (ADR-019)
  - **near-duplicates** — similar numbers in different departments, so retrieval
    has to actually discriminate rather than keyword-match
"""

from __future__ import annotations

import json
import random
from dataclasses import asdict, dataclass
from pathlib import Path

SEED = 20261003
OUT = Path(__file__).resolve().parents[4] / "data" / "raw" / "corpus"

DEPARTMENTS = ["finance", "emea-sales", "apac-sales", "audit", "corp-dev"]
QUARTERS = ["Q1 FY2026", "Q2 FY2026", "Q3 FY2026"]


@dataclass
class PlantedFact:
    """Ground truth. `answer` is what a correct response must contain; `doc_id` and
    `page` are what a correct citation must point at."""

    question: str
    answer: str
    doc_id: str
    doc_version: int
    page: int
    classification: str
    department: str
    min_clearance: str


@dataclass
class Document:
    doc_id: str
    version: int
    title: str
    department: str
    classification: str
    effective_date: str
    superseded_by: int | None
    pages: list[str]


def _money(rng: random.Random, low: int, high: int) -> str:
    return f"EUR {rng.randint(low, high):,}.{rng.randint(0, 99):02d}"


def build() -> tuple[list[Document], list[PlantedFact]]:
    rng = random.Random(SEED)
    docs: list[Document] = []
    facts: list[PlantedFact] = []

    # 1. Quarterly reports: internal, one per quarter per sales region, plus a
    #    restatement of Q2 EMEA so document versioning has a real case.
    for dept in ("emea-sales", "apac-sales"):
        for qi, quarter in enumerate(QUARTERS):
            actual = _money(rng, 900_000, 2_400_000)
            plan = _money(rng, 900_000, 2_400_000)
            doc_id = f"QR-{dept.upper()}-{quarter.replace(' ', '')}"
            body_p1 = (
                f"{quarter} Revenue Report — {dept.replace('-', ' ').title()}\n\n"
                f"Reported revenue for {quarter} was {actual}, against an approved plan of "
                f"{plan}. The variance is attributed to timing of enterprise renewals and "
                f"two large deals closing inside the period rather than after it.\n\n"
                f"Gross margin held at {rng.randint(58, 74)}.{rng.randint(0, 9)}%. "
                f"Headcount closed the quarter at {rng.randint(40, 180)}."
            )
            body_p2 = (
                f"Collections and ageing — {quarter}\n\n"
                f"Days sales outstanding ended the quarter at {rng.randint(38, 72)} days. "
                f"Invoices over 90 days past due totalled {_money(rng, 20_000, 300_000)}, "
                f"concentrated in {rng.randint(2, 9)} accounts.\n\n"
                f"The provision for doubtful debts was increased by "
                f"{_money(rng, 5_000, 60_000)} in the period."
            )
            docs.append(
                Document(
                    doc_id=doc_id,
                    version=1,
                    title=f"{quarter} Revenue Report — {dept}",
                    department=dept,
                    classification="internal",
                    effective_date=f"2026-{(qi + 1) * 3:02d}-28",
                    superseded_by=2 if (dept == "emea-sales" and quarter == "Q2 FY2026") else None,
                    pages=[body_p1, body_p2],
                )
            )
            facts.append(
                PlantedFact(
                    question=f"What was {quarter} revenue for {dept.replace('-', ' ')}?",
                    answer=actual,
                    doc_id=doc_id,
                    doc_version=1,
                    page=1,
                    classification="internal",
                    department=dept,
                    min_clearance="internal",
                )
            )

    # The restatement: same doc_id, version 2, a different number. Both must remain
    # retrievable — the old one for audit, the new one as current.
    restated = _money(rng, 900_000, 2_400_000)
    docs.append(
        Document(
            doc_id="QR-EMEA-SALES-Q2FY2026",
            version=2,
            title="Q2 FY2026 Revenue Report — emea-sales (restated)",
            department="emea-sales",
            classification="internal",
            effective_date="2026-07-15",
            superseded_by=None,
            pages=[
                "Q2 FY2026 Revenue Report — EMEA Sales (RESTATED)\n\n"
                f"Following review, reported revenue for Q2 FY2026 is restated to {restated}. "
                "The original filing recognised two subscription renewals in the wrong period. "
                "This version supersedes version 1 and is the figure to be used for all "
                "external reporting.\n\n"
                "The restatement does not affect full-year guidance."
            ],
        )
    )
    facts.append(
        PlantedFact(
            question="What is the restated Q2 FY2026 revenue for EMEA sales?",
            answer=restated,
            doc_id="QR-EMEA-SALES-Q2FY2026",
            doc_version=2,
            page=1,
            classification="internal",
            department="emea-sales",
            min_clearance="internal",
        )
    )

    # 2. Public: a policy anyone may read. Department-agnostic by design, which is
    #    the one branch of the ACL predicate that ignores department.
    docs.append(
        Document(
            doc_id="POL-EXPENSE-001",
            version=3,
            title="Expense Reimbursement Policy",
            department="finance",
            classification="public",
            effective_date="2026-01-01",
            superseded_by=None,
            pages=[
                "Expense Reimbursement Policy (v3)\n\n"
                "Claims must be submitted within 60 days of the expense being incurred. "
                "Receipts are required for any single item above EUR 25.00. Mileage is "
                "reimbursed at EUR 0.42 per kilometre.\n\n"
                "Approval thresholds: line managers up to EUR 2,000.00; department heads up "
                "to EUR 10,000.00; CFO office above that."
            ],
        )
    )
    facts.append(
        PlantedFact(
            question="What is the receipt threshold for expense claims?",
            answer="EUR 25.00",
            doc_id="POL-EXPENSE-001",
            doc_version=3,
            page=1,
            classification="public",
            department="finance",
            min_clearance="public",
        )
    )

    # 3. Restricted: audit findings, audit department only.
    finding_amount = _money(rng, 80_000, 450_000)
    docs.append(
        Document(
            doc_id="AUD-2026-014",
            version=1,
            title="Internal Audit Finding 2026-014 — Revenue Recognition",
            department="audit",
            classification="restricted",
            effective_date="2026-08-03",
            superseded_by=None,
            pages=[
                "Internal Audit Finding 2026-014 — Revenue Recognition Controls\n\n"
                f"Testing identified {finding_amount} of revenue recognised in the incorrect "
                "period across two EMEA subscription renewals. The control failure was in the "
                "manual period-end cut-off review, which was performed but not evidenced.\n\n"
                "Severity: high. Management response accepted; remediation due 2026-10-31."
            ],
        )
    )
    facts.append(
        PlantedFact(
            question="How much revenue did audit finding 2026-014 identify as misstated?",
            answer=finding_amount,
            doc_id="AUD-2026-014",
            doc_version=1,
            page=1,
            classification="restricted",
            department="audit",
            min_clearance="restricted",
        )
    )

    # 4. Confidential: the document that must never leak. Phase 29 attack
    #    simulation #1 and quality gate Q-6 both target exactly this.
    offer = _money(rng, 40_000_000, 120_000_000)
    docs.append(
        Document(
            doc_id="MA-PROJECT-HELIOS-001",
            version=1,
            title="Project Helios — Indicative Offer",
            department="corp-dev",
            classification="confidential",
            effective_date="2026-09-20",
            superseded_by=None,
            pages=[
                "Project Helios — Indicative Offer (CONFIDENTIAL)\n\n"
                f"The indicative offer for the target is {offer}, subject to confirmatory "
                "diligence. Funding is proposed as 60% cash and 40% equity.\n\n"
                "Distribution is limited to the corporate development team and the CFO "
                "office. This document must not be referenced in any system accessible to "
                "the wider organisation."
            ],
        )
    )
    facts.append(
        PlantedFact(
            question="What is the indicative offer for Project Helios?",
            answer=offer,
            doc_id="MA-PROJECT-HELIOS-001",
            doc_version=1,
            page=1,
            classification="confidential",
            department="corp-dev",
            min_clearance="confidential",
        )
    )

    # 5. Near-duplicates: same shape, same quarter, different departments and
    #    numbers. Keyword matching alone cannot separate these, which is the point.
    for dept in ("finance", "emea-sales"):
        amount = _money(rng, 100_000, 400_000)
        doc_id = f"BUD-{dept.upper()}-FY2026"
        docs.append(
            Document(
                doc_id=doc_id,
                version=1,
                title=f"FY2026 Budget Memo — {dept}",
                department=dept,
                classification="internal",
                effective_date="2026-02-10",
                superseded_by=None,
                pages=[
                    f"FY2026 Budget Memo — {dept.replace('-', ' ').title()}\n\n"
                    f"The approved discretionary budget for FY2026 is {amount}. Travel is "
                    f"capped at {_money(rng, 10_000, 60_000)} and software at "
                    f"{_money(rng, 20_000, 90_000)}.\n\n"
                    "Reallocation between lines above 10% requires CFO office approval."
                ],
            )
        )
        facts.append(
            PlantedFact(
                question=f"What is the FY2026 discretionary budget for {dept.replace('-', ' ')}?",
                answer=amount,
                doc_id=doc_id,
                doc_version=1,
                page=1,
                classification="internal",
                department=dept,
                min_clearance="internal",
            )
        )

    return docs, facts


def write() -> tuple[Path, int, int]:
    docs, facts = build()
    OUT.mkdir(parents=True, exist_ok=True)
    for d in docs:
        (OUT / f"{d.doc_id}.v{d.version}.json").write_text(
            json.dumps(asdict(d), indent=2), encoding="utf-8"
        )
    eval_dir = OUT.parents[1] / "evaluation"
    eval_dir.mkdir(parents=True, exist_ok=True)
    (eval_dir / "planted_facts.json").write_text(
        json.dumps([asdict(f) for f in facts], indent=2), encoding="utf-8"
    )
    return OUT, len(docs), len(facts)


@dataclass
class HardQuestion:
    """A question chosen to expose where one retrieval strategy fails.

    Phase 8 measured Recall@3 = 1.000 with dense retrieval alone on the planted
    facts, which leaves no headroom to demonstrate that hybrid or reranking help.
    These are deliberately harder, and each is labelled with *why* it is hard, so
    a configuration comparison shows a mechanism rather than noise.
    """

    question: str
    expect_doc_id: str
    expect_version: int | None  # None = any version acceptable
    hardness: str  # lexical | paraphrase | near-duplicate | version | distractor
    rationale: str


def hard_questions() -> list[HardQuestion]:
    docs, _ = build()
    by_id = {(d.doc_id, d.version): d for d in docs}

    q: list[HardQuestion] = [
        # Exact identifiers: dense embeddings compress codes into near-identical
        # vectors, so BM25 should win these outright.
        HardQuestion(
            question="AUD-2026-014",
            expect_doc_id="AUD-2026-014",
            expect_version=1,
            hardness="lexical",
            rationale="bare document code, no natural language; dense has nothing to embed",
        ),
        HardQuestion(
            question="What does POL-EXPENSE-001 require?",
            expect_doc_id="POL-EXPENSE-001",
            expect_version=3,
            hardness="lexical",
            rationale="identifier plus generic verb; the code carries all the signal",
        ),
        HardQuestion(
            question="BUD-FINANCE-FY2026 travel cap",
            expect_doc_id="BUD-FINANCE-FY2026",
            expect_version=1,
            hardness="lexical",
            rationale="identifier disambiguates two otherwise identical budget memos",
        ),
        # Paraphrase with no shared vocabulary: BM25 scores ~0, dense should win.
        HardQuestion(
            question="How long do staff have to claim money back after spending it?",
            expect_doc_id="POL-EXPENSE-001",
            expect_version=3,
            hardness="paraphrase",
            rationale="no term overlap with 'Expense Reimbursement Policy' or '60 days'",
        ),
        HardQuestion(
            question="Are customers taking too long to settle their bills?",
            expect_doc_id="QR-EMEA-SALES-Q1FY2026",
            expect_version=1,
            hardness="paraphrase",
            rationale="asks about DSO and ageing without using either term",
        ),
        HardQuestion(
            question="Did anyone book income in the wrong months?",
            expect_doc_id="AUD-2026-014",
            expect_version=1,
            hardness="paraphrase",
            rationale="'revenue recognised in the incorrect period' with every term replaced",
        ),
        # Near-duplicates: two documents of identical shape, different department.
        HardQuestion(
            question="What is the discretionary budget for the EMEA sales team?",
            expect_doc_id="BUD-EMEA-SALES-FY2026",
            expect_version=1,
            hardness="near-duplicate",
            rationale="BUD-FINANCE-FY2026 is near-identical except for department",
        ),
        HardQuestion(
            question="What is the finance department's discretionary budget?",
            expect_doc_id="BUD-FINANCE-FY2026",
            expect_version=1,
            hardness="near-duplicate",
            rationale="mirror of the previous question; both must resolve correctly",
        ),
        # Version disambiguation: same doc_id, two versions, different figures.
        HardQuestion(
            question=(
                "What was originally reported for Q2 FY2026 EMEA revenue, before any restatement?"
            ),
            expect_doc_id="QR-EMEA-SALES-Q2FY2026",
            expect_version=1,
            hardness="version",
            rationale="v2 supersedes v1 and scores higher on most terms; v1 is correct here",
        ),
        HardQuestion(
            question="What is the corrected Q2 FY2026 EMEA revenue figure after review?",
            expect_doc_id="QR-EMEA-SALES-Q2FY2026",
            expect_version=2,
            hardness="version",
            rationale="the mirror case; 'corrected' and 'after review' must select v2",
        ),
        # Distractors: three quarters of the same report shape per region.
        HardQuestion(
            question="APAC sales gross margin in the third quarter",
            expect_doc_id="QR-APAC-SALES-Q3FY2026",
            expect_version=1,
            hardness="distractor",
            rationale="five sibling reports share almost all vocabulary",
        ),
        HardQuestion(
            question="How many accounts hold the overdue balance for EMEA in Q3?",
            expect_doc_id="QR-EMEA-SALES-Q3FY2026",
            expect_version=1,
            hardness="distractor",
            rationale="the answer is on page 2, which the title does not describe",
        ),
    ]

    missing = [h for h in q if (h.expect_doc_id, h.expect_version or 1) not in by_id]
    if missing:
        raise AssertionError(f"hard questions reference documents not in the corpus: {missing}")
    return q


def write_hard_questions() -> Path:
    eval_dir = OUT.parents[1] / "evaluation"
    eval_dir.mkdir(parents=True, exist_ok=True)
    path = eval_dir / "hard_questions.json"
    path.write_text(json.dumps([asdict(h) for h in hard_questions()], indent=2), encoding="utf-8")
    return path


if __name__ == "__main__":
    out, n_docs, n_facts = write()
    print(f"wrote {n_docs} documents to {out}")
    print(f"wrote {n_facts} planted facts to {out.parents[1] / 'evaluation'}")

    _, facts = build()
    by_class: dict[str, int] = {}
    for f in facts:
        by_class[f.classification] = by_class.get(f.classification, 0) + 1
    print()
    print("planted facts by classification:")
    for k, v in sorted(by_class.items()):
        print(f"  {k:<14} {v}")

    hard_path = write_hard_questions()
    hard = hard_questions()
    print()
    print(f"wrote {len(hard)} hard questions to {hard_path}")
    by_hardness: dict[str, int] = {}
    for h in hard:
        by_hardness[h.hardness] = by_hardness.get(h.hardness, 0) + 1
    for k, v in sorted(by_hardness.items()):
        print(f"  {k:<16} {v}")
