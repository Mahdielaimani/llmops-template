"""Smoke tests against the running Compose `core` profile.

Run: uv run poe up && uv run poe test-integration
"""

from __future__ import annotations

import os

import httpx
import pytest

pytestmark = pytest.mark.integration

BASE = os.getenv("LLMOPS_TEST_BASE_URL", "http://localhost:8080")


@pytest.fixture(scope="module")
def http() -> httpx.Client:
    with httpx.Client(base_url=BASE, timeout=5.0) as c:
        try:
            c.get("/health/live")
        except httpx.ConnectError:
            pytest.skip(f"stack not running at {BASE}")
        yield c


def test_live(http: httpx.Client) -> None:
    r = http.get("/health/live")
    assert r.status_code == 200
    assert r.json()["status"] == "ok"
    assert "x-request-id" in r.headers


def test_ready(http: httpx.Client) -> None:
    r = http.get("/health/ready")
    assert r.status_code == 200, r.text


def test_request_id_roundtrip(http: httpx.Client) -> None:
    r = http.get("/health/live", headers={"x-request-id": "it-42"})
    assert r.headers["x-request-id"] == "it-42"


def test_404_envelope(http: httpx.Client) -> None:
    r = http.get("/nothing-here")
    assert r.status_code == 404
    assert r.json()["error"]["code"] == "not_found"


def test_stateful_services_reachable() -> None:
    """Direct pings prove service discovery by Compose DNS is not needed from the host,
    but the published ports are (used by later phases' tooling)."""
    q = httpx.get("http://localhost:6333/healthz", timeout=3.0)
    assert q.status_code == 200


def test_chat_roundtrip(http: httpx.Client) -> None:
    r = http.post(
        "/chat",
        json={
            "messages": [{"role": "user", "content": "what was q2 emea revenue versus plan"}],
            "max_tokens": 6,
        },
        headers={"x-request-id": "it-chat"},
    )
    assert r.status_code == 200
    b = r.json()
    assert b["request_id"] == "it-chat"
    assert b["output_tokens"] == 6  # max_tokens caps a reply that would be longer
    assert b["total_tokens"] == b["input_tokens"] + 6
    assert b["grounded"] is False


def test_chat_streams_over_the_wire(http: httpx.Client) -> None:
    """TestClient can fake streaming; only a real socket proves the response is not buffered."""
    with http.stream(
        "POST",
        "/chat",
        json={"messages": [{"role": "user", "content": "stream me"}], "stream": True},
    ) as r:
        assert r.status_code == 200
        assert r.headers["content-type"].startswith("text/event-stream")
        events = [line for line in r.iter_lines() if line.startswith("event:")]
    assert events[-1] == "event: done"
    assert events.count("event: delta") > 1


CLASSICAL_BASE = os.getenv("LLMOPS_TEST_CLASSICAL_URL", "http://localhost:8090")
_INVOICE = {
    "amount_eur": 24500.0,
    "days_to_due": 30,
    "customer_prior_invoices": 12,
    "customer_prior_late_ratio": 0.25,
    "has_purchase_order": True,
    "is_new_customer": False,
}


@pytest.fixture(scope="module")
def ml(http: httpx.Client) -> httpx.Client:
    with httpx.Client(base_url=CLASSICAL_BASE, timeout=5.0) as c:
        try:
            c.get("/health/live")
        except httpx.ConnectError:
            pytest.skip(f"classical-ml not running at {CLASSICAL_BASE}")
        yield c


def test_classical_predict(ml: httpx.Client) -> None:
    r = ml.post("/predict", json=_INVOICE, headers={"x-request-id": "it-ml"})
    assert r.status_code == 200
    b = r.json()
    assert b["request_id"] == "it-ml"
    assert 0.0 <= b["probability"] <= 1.0
    assert b["model_version"] == "1.0.0"


def test_classical_reports_its_own_latency(ml: httpx.Client) -> None:
    """Phase 3: host round-trip includes ~43ms of Docker Desktop port proxy, so the
    handler's own measurement is the only usable latency signal from the host."""
    b = ml.post("/predict", json=_INVOICE).json()
    assert b["latency_ms"] < 20  # in-handler; the round-trip would be ~44ms


def test_classical_artifact_provenance(ml: httpx.Client) -> None:
    card = ml.get("/model").json()
    assert len(card["artifact_sha256"]) == 64
    assert card["metrics"]["roc_auc"] > card["metrics"].get("baseline_accuracy", 0) - 1
    assert card["metrics"]["accuracy"] > card["metrics"]["baseline_accuracy"]
