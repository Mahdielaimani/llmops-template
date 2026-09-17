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
