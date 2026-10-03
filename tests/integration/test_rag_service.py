"""Live rag-service. The ACL cases are the reason this file exists: a unit test
proves the predicate is right, only an HTTP round-trip proves the wiring is.

    docker compose --profile core --profile rag up -d --build
    uv run poe test-integration
"""

from __future__ import annotations

import os

import httpx
import pytest

pytestmark = pytest.mark.integration

RAG = os.getenv("LLMOPS_TEST_RAG_URL", "http://localhost:8070")
HELIOS = "MA-PROJECT-HELIOS-001"
HELIOS_Q = {"question": "What is the indicative offer for Project Helios?", "top_k": 5}


@pytest.fixture(scope="module")
def rag() -> httpx.Client:
    with httpx.Client(base_url=RAG, timeout=20.0) as c:
        try:
            c.get("/health/live")
        except httpx.ConnectError:
            pytest.skip(f"rag-service not running at {RAG}")
        yield c


def _doc_ids(payload: dict) -> list[str]:
    return [c["doc_id"] for c in payload["citations"]]


def test_readiness_checks_qdrant_for_real(rag: httpx.Client) -> None:
    """Phase 1 shipped an empty HealthRegistry; this is the first service that
    registers a real dependency check, so readiness finally means something."""
    body = rag.get("/health/ready").json()
    names = {c["name"] for c in body["checks"]}
    assert "qdrant" in names
    assert all(c["status"] == "ok" for c in body["checks"])


def test_index_is_identified_by_its_inputs(rag: httpx.Client) -> None:
    body = rag.get("/index").json()
    assert body["collection"] == f"docs_{body['index_version']}"
    assert body["embed_dim"] == 384
    assert body["chunks"] > 0


@pytest.mark.parametrize(
    ("user", "may_see_helios"),
    [
        ("contractor", False),
        ("analyst-emea", False),
        ("analyst-apac", False),
        ("auditor", False),
        ("cfo-office", True),
    ],
)
def test_confidential_memo_reaches_only_the_cleared_identity(
    rag: httpx.Client, user: str, may_see_helios: bool
) -> None:
    """Quality gate Q-6, over HTTP."""
    body = rag.post("/rag", json=HELIOS_Q, headers={"x-user": user}).json()
    assert (HELIOS in _doc_ids(body)) is may_see_helios


def test_candidate_counts_differ_by_clearance(rag: httpx.Client) -> None:
    """The pre-filter working: disallowed chunks were never candidates, so top_k is
    honoured within each user's permitted corpus rather than collapsing after a
    post-filter (ADR-019)."""
    counts = {
        u: rag.post("/rag", json=HELIOS_Q, headers={"x-user": u}).json()["candidates_considered"]
        for u in ("contractor", "auditor", "cfo-office")
    }
    assert counts["contractor"] < counts["auditor"] < counts["cfo-office"]


def test_expired_identity_retrieves_nothing(rag: httpx.Client) -> None:
    body = rag.post(
        "/rag",
        json={"question": "audit finding revenue recognition", "top_k": 5},
        headers={"x-user": "auditor-expired"},
    ).json()
    assert body["candidates_considered"] == 0
    assert body["citations"] == []


def test_unknown_identity_fails_closed(rag: httpx.Client) -> None:
    """No header, and a bogus header, both resolve to the least-privileged user."""
    anonymous = rag.post("/rag", json=HELIOS_Q).json()
    bogus = rag.post("/rag", json=HELIOS_Q, headers={"x-user": "root"}).json()
    contractor = rag.post("/rag", json=HELIOS_Q, headers={"x-user": "contractor"}).json()
    assert anonymous["acl_scope"] == bogus["acl_scope"] == contractor["acl_scope"]
    assert HELIOS not in _doc_ids(anonymous)


def test_acl_scope_differs_per_identity(rag: httpx.Client) -> None:
    """Phase 15 must put this in the cache key, or a lower clearance hits a higher
    clearance's cached answer (ADR-019)."""
    scopes = {
        rag.get("/whoami", headers={"x-user": u}).json()["acl_scope_hash"]
        for u in ("contractor", "analyst-emea", "analyst-apac", "auditor", "cfo-office")
    }
    assert len(scopes) == 5


def test_citations_are_traceable_to_a_document_version(rag: httpx.Client) -> None:
    body = rag.post(
        "/rag",
        json={"question": "What was Q2 FY2026 revenue for emea sales?", "top_k": 5},
        headers={"x-user": "analyst-emea"},
    ).json()
    assert body["citations"], "expected at least one citation"
    first = body["citations"][0]
    assert first["doc_version"] >= 1
    assert f"@v{first['doc_version']}" in first["citation"]
    assert f"p.{first['page']}" in first["citation"]


def test_answer_is_null_and_says_so(rag: httpx.Client) -> None:
    """Phase 8 is retrieval. Returning a fabricated answer to look complete would
    be worse than returning null; generation lands with the gateway in Phase 15."""
    body = rag.post("/rag", json=HELIOS_Q, headers={"x-user": "cfo-office"}).json()
    assert body["answer"] is None
    assert body["grounded"] is True
    assert body["context_chars"] > 0


def test_request_id_and_timings_are_reported(rag: httpx.Client) -> None:
    body = rag.post(
        "/rag", json=HELIOS_Q, headers={"x-user": "cfo-office", "x-request-id": "it-rag"}
    ).json()
    assert body["request_id"] == "it-rag"
    assert body["embed_ms"] > 0
    assert body["retrieval_ms"] >= body["embed_ms"]


def test_validation_rejects_empty_question(rag: httpx.Client) -> None:
    r = rag.post("/rag", json={"question": "", "top_k": 5}, headers={"x-user": "contractor"})
    assert r.status_code == 422
    assert r.json()["error"]["code"] == "validation_error"
