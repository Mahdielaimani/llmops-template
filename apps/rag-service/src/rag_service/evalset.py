"""Generate the evaluation dataset.

    uv run python -m rag_service.evalset

Phase 9 compared five retrieval configurations on 12 questions and recorded the
problem honestly: one question is worth 0.083 MRR, so differences below ~0.15
were indistinguishable from noise. This module exists to remove that excuse.

Questions are **generated from the corpus**, not written by hand, for three
reasons: the answers are derived from the same seeded data so ground truth
cannot drift, the set is reproducible from a content hash (ADR-015 class 8), and
coverage per hardness class is countable rather than hoped for.

Each item carries what makes it scorable:
  - `expect_doc_id` / `expect_version` — retrieval ground truth
  - `expect_answer` — substring the generated answer must contain (Phase 15)
  - `hardness` — which failure mode the question targets
  - `min_clearance` / `department` — which identities *should* be able to answer,
    so the same set measures ACL enforcement rather than needing a second one
"""

from __future__ import annotations

import hashlib
import json
from dataclasses import asdict, dataclass
from pathlib import Path

from rag_service.corpus import OUT, build, hard_questions

EVAL_DIR = OUT.parents[1] / "evaluation"
DATASET_VERSION = "1.0.0"


@dataclass
class EvalItem:
    id: str
    question: str
    expect_doc_id: str
    expect_version: int | None
    expect_answer: str | None
    hardness: str
    min_clearance: str
    department: str
    rationale: str = ""


def _templates(doc_id: str, title: str, quarter: str | None) -> list[tuple[str, str, str]]:
    """(question, hardness, rationale) variants for one document.

    Several phrasings per document is the cheapest way to multiply the set without
    inventing facts: every variant has the same ground truth, so a configuration
    that only handles one phrasing is visibly worse than one that handles all.
    """
    short = title.split(" — ")[0]
    out: list[tuple[str, str, str]] = [
        (doc_id, "lexical", "bare document code; dense has nothing to embed"),
        (f"What does {doc_id} contain?", "lexical", "identifier plus a generic verb"),
        (f"Summarise {short}", "distractor", "title words are shared across siblings"),
    ]
    if quarter:
        out += [
            (f"What was {quarter} revenue?", "distractor", "six sibling reports share vocabulary"),
            (
                f"How did {quarter} compare against plan?",
                "paraphrase",
                "asks for the variance without using the word",
            ),
            (
                f"What were the collections figures in {quarter}?",
                "distractor",
                "the answer is on page 2, which the title does not describe",
            ),
        ]
    return out


def build_dataset() -> list[EvalItem]:
    docs, facts = build()
    items: list[EvalItem] = []
    seen: set[str] = set()

    def add(
        question: str,
        doc_id: str,
        version: int | None,
        answer: str | None,
        hardness: str,
        clearance: str,
        department: str,
        rationale: str = "",
    ) -> None:
        key = question.strip().lower()
        if key in seen:
            return
        seen.add(key)
        items.append(
            EvalItem(
                id=f"q{len(items) + 1:03d}",
                question=question,
                expect_doc_id=doc_id,
                expect_version=version,
                expect_answer=answer,
                hardness=hardness,
                min_clearance=clearance,
                department=department,
                rationale=rationale,
            )
        )

    # 1. The planted facts: exact-answer ground truth, the only items with an
    #    `expect_answer` a generated answer can be checked against.
    for f in facts:
        add(
            f["question"] if isinstance(f, dict) else f.question,
            f["doc_id"] if isinstance(f, dict) else f.doc_id,
            f["doc_version"] if isinstance(f, dict) else f.doc_version,
            f["answer"] if isinstance(f, dict) else f.answer,
            "planted",
            f["min_clearance"] if isinstance(f, dict) else f.min_clearance,
            f["department"] if isinstance(f, dict) else f.department,
            "exact figure planted at a known location",
        )

    # 2. The hand-written hard questions from Phase 9, kept so the two sets are
    #    comparable and the Phase 9 result can be re-checked on the same items.
    doc_meta = {(d.doc_id, d.version): d for d in docs}
    for h in hard_questions():
        meta = doc_meta[(h.expect_doc_id, h.expect_version or 1)]
        add(
            h.question,
            h.expect_doc_id,
            h.expect_version,
            None,
            h.hardness,
            meta.classification,
            meta.department,
            h.rationale,
        )

    # 3. Generated variants, several phrasings per document.
    for d in docs:
        quarter = None
        for q in ("Q1 FY2026", "Q2 FY2026", "Q3 FY2026"):
            if q.replace(" ", "") in d.doc_id or q in d.title:
                quarter = q
                break
        for question, hardness, rationale in _templates(d.doc_id, d.title, quarter):
            add(
                question,
                d.doc_id,
                d.version,
                None,
                hardness,
                d.classification,
                d.department,
                rationale,
            )

    # 4. Cross-department near-duplicates: the same question shape aimed at each
    #    department, so a retriever that ignores the qualifier is visibly wrong.
    for d in docs:
        if d.doc_id.startswith("BUD-"):
            dept = d.department.replace("-", " ")
            add(
                f"What is the {dept} discretionary budget for FY2026?",
                d.doc_id,
                d.version,
                None,
                "near-duplicate",
                d.classification,
                d.department,
                "a sibling memo is identical except for department",
            )
            add(
                f"How much can {dept} spend on travel?",
                d.doc_id,
                d.version,
                None,
                "near-duplicate",
                d.classification,
                d.department,
                "same, phrased without the word budget",
            )

    # 4b. Paraphrase and version variants, targeted at the classes that actually
    #     discriminate between configurations. Phase 9 showed `lexical` questions
    #     are easy for BM25 and `near-duplicate` easy for everything, so adding
    #     more of those would inflate n without improving resolution.
    paraphrases = [
        ("Are customers paying slowly?", "days sales outstanding", "page 2, no shared terms"),
        ("How much is tied up in old unpaid invoices?", "over 90 days past due", "page 2"),
        ("Did we set aside more money for bad debt?", "provision for doubtful debts", "page 2"),
        ("How profitable were sales?", "gross margin", "page 1, synonym only"),
        ("How many people were on the team?", "headcount", "page 1, synonym only"),
        ("Did we hit our targets?", "against an approved plan", "page 1, variance paraphrased"),
    ]
    for d in docs:
        if not d.doc_id.startswith("QR-"):
            continue
        dept = d.department.replace("-", " ")
        quarter = next(
            (q for q in ("Q1 FY2026", "Q2 FY2026", "Q3 FY2026") if q.replace(" ", "") in d.doc_id),
            None,
        )
        if quarter is None:
            continue
        for question, _target, rationale in paraphrases:
            add(
                f"{question} ({dept}, {quarter})",
                d.doc_id,
                d.version,
                None,
                "paraphrase",
                d.classification,
                d.department,
                rationale,
            )

    # 4c. Version disambiguation. Dense retrieval was weakest here in Phase 9
    #     (MRR 0.375), and the corpus has exactly one restated document, so each
    #     phrasing has to resolve to the right side of the restatement.
    versioned = [
        ("What was first reported for {q} {dept} revenue?", 1),
        ("What did the original {q} {dept} filing say?", 1),
        ("What was {q} {dept} revenue before the correction?", 1),
        ("What is the restated {q} {dept} revenue?", 2),
        ("What is the corrected {q} {dept} revenue after review?", 2),
        ("Which figure should be used externally for {q} {dept}?", 2),
        ("What is the current {q} {dept} revenue figure?", 2),
    ]
    for template, version in versioned:
        add(
            template.format(q="Q2 FY2026", dept="EMEA sales"),
            "QR-EMEA-SALES-Q2FY2026",
            version,
            None,
            "version",
            "internal",
            "emea-sales",
            f"must resolve to v{version} of a document with two versions",
        )

    # 5. Unanswerable questions. The correct behaviour is abstention, and a system
    #    that always answers scores worse here than one that sometimes declines
    #    (agent-platform.md §4: coverage against accuracy).
    for question in (
        "What was Q4 FY2027 revenue for the LATAM region?",
        "Who is the chief executive?",
        "What is the share price today?",
        "How many employees are in the Tokyo office?",
        "What is the dividend policy?",
    ):
        add(
            question,
            "__none__",
            None,
            "insufficient_evidence",
            "unanswerable",
            "public",
            "finance",
            "nothing in the corpus answers this; abstention is correct",
        )

    return items


def dataset_sha(items: list[EvalItem]) -> str:
    """A metric is meaningless without the dataset version it was measured on
    (ADR-015 class 8), so the content is hashed rather than trusting a filename."""
    payload = json.dumps([asdict(i) for i in items], sort_keys=True)
    return hashlib.sha256(payload.encode()).hexdigest()[:12]


def write() -> tuple[Path, int, str]:
    items = build_dataset()
    EVAL_DIR.mkdir(parents=True, exist_ok=True)
    sha = dataset_sha(items)
    path = EVAL_DIR / "dataset.json"
    path.write_text(
        json.dumps(
            {
                "version": DATASET_VERSION,
                "sha256": sha,
                "n": len(items),
                "items": [asdict(i) for i in items],
            },
            indent=2,
        ),
        encoding="utf-8",
    )
    return path, len(items), sha


if __name__ == "__main__":
    path, n, sha = write()
    items = build_dataset()
    print(f"wrote {n} items to {path}")
    print(f"dataset {DATASET_VERSION} sha256 {sha}")

    by: dict[str, int] = {}
    for i in items:
        by[i.hardness] = by.get(i.hardness, 0) + 1
    print()
    print("by hardness class:")
    for k, v in sorted(by.items(), key=lambda kv: -kv[1]):
        print(f"  {k:<16} {v:>4}")

    by_clear: dict[str, int] = {}
    for i in items:
        by_clear[i.min_clearance] = by_clear.get(i.min_clearance, 0) + 1
    print()
    print("by minimum clearance required:")
    for k, v in sorted(by_clear.items()):
        print(f"  {k:<16} {v:>4}")

    answerable = sum(1 for i in items if i.expect_doc_id != "__none__")
    print()
    print(f"answerable {answerable}   unanswerable {len(items) - answerable}")
    print(f"with an exact expected answer: {sum(1 for i in items if i.expect_answer)}")
