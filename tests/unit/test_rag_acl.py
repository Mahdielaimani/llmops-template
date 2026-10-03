"""ACL enforcement. These are the hard-fail tests behind quality gate Q-6
(zero unauthorized retrievals) in docs/business-requirements.md §4.

They run without Qdrant: the predicate renders to a plain dict, which is exactly
why `AclPredicate.qdrant_filter()` returns one instead of importing qdrant models.
"""

from __future__ import annotations

from datetime import date, timedelta

import pytest

from rag_service.acl import AclPredicate, Classification, User, payload_of
from rag_service.acl import Chunk as AclChunk


def _payload(classification: str, department: str, doc_id: str = "D1") -> dict[str, object]:
    return payload_of(
        AclChunk(
            chunk_id=f"{doc_id}.c0",
            doc_id=doc_id,
            doc_version=1,
            title="t",
            text="x",
            page=1,
            classification=classification,
            department=department,
            effective_date="2026-01-01",
        )
    )


ANALYST = User(
    user_id="a", clearance=Classification.INTERNAL, departments=frozenset({"emea-sales"})
)
AUDITOR = User(
    user_id="b",
    clearance=Classification.RESTRICTED,
    departments=frozenset({"audit"}),
    valid_from=date.today() - timedelta(days=10),
    valid_until=date.today() + timedelta(days=10),
)
CFO = User(
    user_id="c",
    clearance=Classification.CONFIDENTIAL,
    departments=frozenset({"corp-dev", "emea-sales", "audit"}),
)
CONTRACTOR = User(user_id="d", clearance=Classification.PUBLIC, departments=frozenset())


def test_classification_is_ordered() -> None:
    """The whole rule is `doc <= user`, so the ordering is load-bearing."""
    assert Classification.PUBLIC < Classification.INTERNAL
    assert Classification.INTERNAL < Classification.RESTRICTED
    assert Classification.RESTRICTED < Classification.CONFIDENTIAL


def test_confidential_is_denied_to_everyone_below_it() -> None:
    """Q-6: zero unauthorized retrievals. The M&A memo is the document in the corpus
    that must never leak."""
    doc = _payload("confidential", "corp-dev", "MA-PROJECT-HELIOS-001")
    for user in (CONTRACTOR, ANALYST, AUDITOR):
        assert AclPredicate.for_user(user).allows(doc) is False
    assert AclPredicate.for_user(CFO).allows(doc) is True


def test_clearance_alone_is_not_enough_department_must_match() -> None:
    """ABAC, not RBAC: an internal clearance does not grant every internal document."""
    acl = AclPredicate.for_user(ANALYST)
    assert acl.allows(_payload("internal", "emea-sales")) is True
    assert acl.allows(_payload("internal", "apac-sales")) is False


def test_public_documents_ignore_department() -> None:
    """The one branch of the predicate that skips the department check."""
    for user in (CONTRACTOR, ANALYST, AUDITOR, CFO):
        assert AclPredicate.for_user(user).allows(_payload("public", "finance")) is True


def test_expired_user_is_denied_everything() -> None:
    """Time-boxed access is why this is ABAC: a role cannot express a validity window."""
    expired = User(
        user_id="e",
        clearance=Classification.CONFIDENTIAL,
        departments=frozenset({"corp-dev"}),
        valid_until=date.today() - timedelta(days=1),
    )
    acl = AclPredicate.for_user(expired)
    assert acl.expired is True
    assert acl.allows(_payload("public", "finance")) is False
    assert acl.allows(_payload("confidential", "corp-dev")) is False


def test_not_yet_valid_user_is_denied() -> None:
    future = User(
        user_id="f",
        clearance=Classification.RESTRICTED,
        departments=frozenset({"audit"}),
        valid_from=date.today() + timedelta(days=5),
    )
    assert AclPredicate.for_user(future).allows(_payload("restricted", "audit")) is False


def test_unlabelled_document_fails_closed() -> None:
    """An unlabelled document is not a public one. ADR-019: deny is the default."""
    acl = AclPredicate.for_user(CFO)
    assert acl.allows({"doc_id": "X"}) is False
    assert acl.allows({}) is False


def test_document_scope_narrows_further() -> None:
    """An external auditor scoped to three documents, which is the case roles
    cannot express."""
    scoped = User(
        user_id="g",
        clearance=Classification.RESTRICTED,
        departments=frozenset({"audit"}),
        scoped_doc_ids=frozenset({"AUD-2026-014"}),
    )
    acl = AclPredicate.for_user(scoped)
    assert acl.allows(_payload("restricted", "audit", "AUD-2026-014")) is True
    assert acl.allows(_payload("restricted", "audit", "AUD-2026-099")) is False


# --------------------------------------------------------------- the Qdrant filter


def test_filter_only_admits_levels_at_or_below_clearance() -> None:
    f = AclPredicate.for_user(ANALYST).qdrant_filter()
    levels = next(
        c["match"]["any"]
        for c in f["must"]
        if c.get("key") == "classification_level" and "any" in c["match"]
    )
    assert sorted(levels) == [0, 1]  # public, internal
    assert int(Classification.CONFIDENTIAL) not in levels


def test_filter_for_expired_user_matches_nothing() -> None:
    """An impossible condition rather than an absent filter: a caller that forgets
    to check `expired` must still retrieve zero rows."""
    expired = User(
        user_id="h",
        clearance=Classification.CONFIDENTIAL,
        departments=frozenset({"corp-dev"}),
        valid_until=date.today() - timedelta(days=1),
    )
    f = AclPredicate.for_user(expired).qdrant_filter()
    assert f["must"][0]["match"]["value"] == -1


def test_filter_without_departments_admits_public_only() -> None:
    f = AclPredicate.for_user(CONTRACTOR).qdrant_filter()
    assert any(
        c.get("key") == "classification_level" and c["match"].get("value") == 0 for c in f["must"]
    )


def test_filter_and_predicate_agree(
    # Every combination in the corpus, checked against both enforcement points.
) -> None:
    """The redundancy in retriever.search() is only valuable if the two agree. A
    disagreement would mean a chunk passes the query filter and fails the re-check,
    which the retriever logs as a bug rather than silently dropping."""
    for user in (CONTRACTOR, ANALYST, AUDITOR, CFO):
        acl = AclPredicate.for_user(user)
        allowed_levels = {int(c) for c in Classification if c <= acl.clearance}
        for classification in ("public", "internal", "restricted", "confidential"):
            for department in ("emea-sales", "apac-sales", "audit", "corp-dev", "finance"):
                payload = _payload(classification, department)
                by_predicate = acl.allows(payload)
                level_ok = payload["classification_level"] in allowed_levels
                dept_ok = classification == "public" or department in acl.departments
                assert by_predicate == (level_ok and dept_ok)


# ------------------------------------------------------------------- cache scoping


def test_acl_scope_hash_differs_per_permission_set() -> None:
    """ADR-019: the cache key must include this, or user B at a lower clearance hits
    user A's cached answer about a document B cannot see."""
    scopes = {AclPredicate.for_user(u).scope_hash() for u in (CONTRACTOR, ANALYST, AUDITOR, CFO)}
    assert len(scopes) == 4


def test_acl_scope_hash_is_stable_for_the_same_permissions() -> None:
    a = User(user_id="x", clearance=Classification.INTERNAL, departments=frozenset({"finance"}))
    b = User(user_id="y", clearance=Classification.INTERNAL, departments=frozenset({"finance"}))
    assert AclPredicate.for_user(a).scope_hash() == AclPredicate.for_user(b).scope_hash()


def test_expired_scope_hash_differs_from_valid() -> None:
    valid = User(user_id="x", clearance=Classification.RESTRICTED, departments=frozenset({"audit"}))
    expired = User(
        user_id="x",
        clearance=Classification.RESTRICTED,
        departments=frozenset({"audit"}),
        valid_until=date.today() - timedelta(days=1),
    )
    assert AclPredicate.for_user(valid).scope_hash() != AclPredicate.for_user(expired).scope_hash()


@pytest.mark.parametrize("raw", ["INTERNAL", "internal", " Internal "])
def test_classification_parsing_is_forgiving_of_case_and_space(raw: str) -> None:
    assert Classification.parse(raw) is Classification.INTERNAL
