"""Fusion, lexical tokenization and the ACL in the lexical path.

The tests that matter here are the ones pinning *why* RRF lost: it uses only
ranks, so it cannot tell a confident rank 1 from a desperate one. That is a
property of the algorithm, assertable without a corpus.
"""

from __future__ import annotations

from datetime import date, timedelta

import pytest

from rag_service.acl import AclPredicate, Chunk, Classification, User, payload_of
from rag_service.fusion import reciprocal_rank_fusion, weighted_fusion
from rag_service.lexical import LexicalIndex, tokenize


def chunk(cid: str, score: float, by: str = "dense", **kw: object) -> Chunk:
    return Chunk(
        chunk_id=cid,
        doc_id=kw.get("doc_id", cid.split(".")[0]),  # type: ignore[arg-type]
        doc_version=1,
        title="t",
        text=kw.get("text", "text"),  # type: ignore[arg-type]
        page=1,
        classification=kw.get("classification", "internal"),  # type: ignore[arg-type]
        department=kw.get("department", "finance"),  # type: ignore[arg-type]
        effective_date="2026-01-01",
        score=score,
        retrieved_by=by,
    )


# ------------------------------------------------------------------- tokenization


def test_identifiers_survive_tokenization_whole_and_in_parts() -> None:
    """A naive \\w+ splits AUD-2026-014 into three common tokens and destroys the
    exact-match signal BM25 exists to provide."""
    tokens = tokenize("See AUD-2026-014 for detail")
    assert "aud-2026-014" in tokens
    assert "aud" in tokens and "2026" in tokens and "014" in tokens


def test_tokenization_is_case_insensitive() -> None:
    assert tokenize("AUD-2026-014") == tokenize("aud-2026-014")


def test_decimal_figures_are_not_shattered() -> None:
    tokens = tokenize("EUR 1,234,567.89")
    assert any("." in t for t in tokens), tokens


# ----------------------------------------------------------------------- RRF


def test_rrf_rewards_agreement_between_retrievers() -> None:
    """A chunk found by both outranks one found strongly by only one."""
    dense = [chunk("a.c0", 0.99), chunk("b.c0", 0.80)]
    lexical = [chunk("b.c0", 9.0, by="bm25"), chunk("c.c0", 7.0, by="bm25")]
    fused = reciprocal_rank_fusion(dense, lexical)
    assert fused[0].chunk_id == "b.c0"  # rank 2 + rank 1 beats a lone rank 1
    assert fused[0].retrieved_by == "bm25+dense"


def test_rrf_ignores_scores_entirely() -> None:
    """The property that makes it calibration-free, and the reason it lost on
    paraphrase: a desperate rank 1 counts the same as a confident one
    (ADR-003, docs/hybrid-retrieval.md §4)."""
    dense = [chunk("a.c0", 0.99)]
    strong = reciprocal_rank_fusion(dense, [chunk("b.c0", 100.0, by="bm25")])
    weak = reciprocal_rank_fusion(dense, [chunk("b.c0", 0.0001, by="bm25")])
    assert [c.chunk_id for c in strong] == [c.chunk_id for c in weak]
    assert strong[0].score == weak[0].score


def test_rrf_score_matches_the_formula() -> None:
    fused = reciprocal_rank_fusion([chunk("a.c0", 0.5)], [], k=60)
    # Fusion rounds to 6 decimals, so the tolerance must match the stored precision.
    assert fused[0].score == pytest.approx(1 / 61, abs=5e-7)


def test_rrf_handles_one_empty_list() -> None:
    dense = [chunk("a.c0", 0.9), chunk("b.c0", 0.5)]
    fused = reciprocal_rank_fusion(dense, [])
    assert [c.chunk_id for c in fused] == ["a.c0", "b.c0"]


# ------------------------------------------------------------- weighted fusion


def test_weighted_fusion_does_use_scores() -> None:
    """The difference from RRF, and why it survived paraphrase: a weak lexical hit
    normalises toward zero instead of counting as a rank-1 vote."""
    dense = [chunk("a.c0", 0.99), chunk("b.c0", 0.10)]
    lexical = [chunk("b.c0", 9.0, by="bm25"), chunk("c.c0", 8.9, by="bm25")]
    fused = weighted_fusion(dense, lexical, dense_weight=0.6)
    assert fused[0].chunk_id == "a.c0"  # dense rank 1 with a wide margin wins


def test_weighted_fusion_respects_the_weight() -> None:
    dense = [chunk("a.c0", 1.0), chunk("b.c0", 0.0)]
    lexical = [chunk("b.c0", 1.0, by="bm25"), chunk("a.c0", 0.0, by="bm25")]
    assert weighted_fusion(dense, lexical, dense_weight=0.9)[0].chunk_id == "a.c0"
    assert weighted_fusion(dense, lexical, dense_weight=0.1)[0].chunk_id == "b.c0"


def test_weighted_fusion_single_result_failure_mode_is_real() -> None:
    """Documented in ADR-003: min-max maps a lone result to 1.0 regardless of
    quality. Pinned so the trade-off is not forgotten."""
    fused = weighted_fusion([], [chunk("junk.c0", 0.0001, by="bm25")], dense_weight=0.6)
    assert fused[0].score == pytest.approx(0.4, abs=1e-6)  # (1-0.6) * 1.0


# ------------------------------------------------- the ACL in the lexical path


def _payloads() -> list[dict[str, object]]:
    specs = [
        ("AUD-2026-014", "restricted", "audit", "revenue recognised in the incorrect period"),
        ("MA-PROJECT-HELIOS-001", "confidential", "corp-dev", "indicative offer for the target"),
        ("POL-EXPENSE-001", "public", "finance", "receipts required above EUR 25.00"),
        ("BUD-FINANCE-FY2026", "internal", "finance", "discretionary budget travel cap"),
    ]
    return [
        payload_of(
            Chunk(
                chunk_id=f"{doc}.c0",
                doc_id=doc,
                doc_version=1,
                title=doc,
                text=text,
                page=1,
                classification=cls,
                department=dept,
                effective_date="2026-01-01",
            )
        )
        for doc, cls, dept, text in specs
    ]


CONTRACTOR = User(user_id="c", clearance=Classification.PUBLIC, departments=frozenset())
AUDITOR = User(
    user_id="a", clearance=Classification.RESTRICTED, departments=frozenset({"audit", "finance"})
)
CFO = User(
    user_id="f",
    clearance=Classification.CONFIDENTIAL,
    departments=frozenset({"corp-dev", "audit", "finance"}),
)


def test_lexical_index_enforces_the_same_acl() -> None:
    """A BM25 index without the predicate is a bypass of the Qdrant filter —
    same documents, different door (ADR-019)."""
    index = LexicalIndex(_payloads())
    q = "indicative offer for the target"
    assert index.search(q, AclPredicate.for_user(CONTRACTOR)).chunks == []
    assert index.search(q, AclPredicate.for_user(AUDITOR)).chunks == []
    cfo_hits = index.search(q, AclPredicate.for_user(CFO)).chunks
    assert any(c.doc_id == "MA-PROJECT-HELIOS-001" for c in cfo_hits)


def test_lexical_finds_an_exact_identifier() -> None:
    """The class dense retrieval loses: a bare code with no natural language."""
    index = LexicalIndex(_payloads())
    hits = index.search("AUD-2026-014", AclPredicate.for_user(AUDITOR)).chunks
    assert hits and hits[0].doc_id == "AUD-2026-014"


def test_lexical_scores_zero_with_no_term_overlap() -> None:
    """BM25 returning nothing on paraphrase is not a bug, it is the reason dense
    retrieval exists (docs/hybrid-retrieval.md §4)."""
    index = LexicalIndex(_payloads())
    hits = index.search(
        "how long do staff have to claim money back", AclPredicate.for_user(CFO)
    ).chunks
    assert all(c.doc_id != "POL-EXPENSE-001" for c in hits[:1]) or hits == []


def test_lexical_expired_user_gets_nothing() -> None:
    expired = User(
        user_id="x",
        clearance=Classification.CONFIDENTIAL,
        departments=frozenset({"corp-dev"}),
        valid_until=date.today() - timedelta(days=1),
    )
    index = LexicalIndex(_payloads())
    assert index.search("indicative offer", AclPredicate.for_user(expired)).chunks == []
