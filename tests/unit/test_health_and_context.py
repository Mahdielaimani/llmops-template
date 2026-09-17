from __future__ import annotations

import asyncio

import pytest
from fastapi import FastAPI
from fastapi.testclient import TestClient

from llmops_core.context import REQUEST_ID_HEADER, RequestContextMiddleware, get_request_id
from llmops_core.errors import NotFoundError, install_error_handlers
from llmops_core.health import HealthRegistry, build_health_router


def test_live_and_ready_ok(client: TestClient) -> None:
    live = client.get("/health/live")
    ready = client.get("/health/ready")
    assert live.status_code == 200 and live.json()["status"] == "ok"
    assert ready.status_code == 200 and ready.json()["status"] == "ok"
    assert live.json()["service"] == "llm-application"


def test_request_id_generated_and_echoed(client: TestClient) -> None:
    r = client.get("/health/live")
    rid = r.headers[REQUEST_ID_HEADER]
    assert len(rid) == 32  # uuid4 hex


def test_request_id_propagated_when_supplied(client: TestClient) -> None:
    r = client.get("/health/live", headers={REQUEST_ID_HEADER: "kong-abc-123"})
    assert r.headers[REQUEST_ID_HEADER] == "kong-abc-123"


def test_malformed_request_id_replaced(client: TestClient) -> None:
    r = client.get("/health/live", headers={REQUEST_ID_HEADER: "x" * 500})
    assert r.headers[REQUEST_ID_HEADER] != "x" * 500


def _app_with_checks(registry: HealthRegistry) -> TestClient:
    app = FastAPI()
    app.add_middleware(RequestContextMiddleware)
    install_error_handlers(app)
    app.include_router(build_health_router(registry))

    @app.get("/boom")
    async def boom() -> None:
        raise NotFoundError("thing missing", details={"id": 7})

    @app.get("/crash")
    async def crash() -> None:
        raise RuntimeError("secret internal detail")

    @app.get("/whoami")
    async def whoami() -> dict[str, str | None]:
        return {"request_id": get_request_id()}

    return TestClient(app, raise_server_exceptions=False)


def test_readiness_degraded_on_failing_dependency() -> None:
    reg = HealthRegistry(service="t", default_timeout_s=0.2)

    async def ok() -> None:
        return None

    async def fails() -> None:
        raise ConnectionError("db down")

    async def slow() -> None:
        await asyncio.sleep(1)

    reg.register("ok", ok)
    reg.register("db", fails)
    reg.register("slow", slow)
    c = _app_with_checks(reg)

    r = c.get("/health/ready")
    assert r.status_code == 503
    body = r.json()
    assert body["status"] == "degraded"
    by_name = {x["name"]: x for x in body["checks"]}
    assert by_name["ok"]["status"] == "ok"
    assert by_name["db"]["status"] == "fail" and "db down" in by_name["db"]["error"]
    assert by_name["slow"]["status"] == "timeout"
    # liveness unaffected by dependencies
    assert c.get("/health/live").status_code == 200


def test_error_envelope_shape() -> None:
    c = _app_with_checks(HealthRegistry(service="t"))
    r = c.get("/boom", headers={REQUEST_ID_HEADER: "rid-1"})
    assert r.status_code == 404
    assert r.json() == {
        "error": {
            "code": "not_found",
            "message": "thing missing",
            "request_id": "rid-1",
            "details": {"id": 7},
        }
    }


def test_unhandled_exception_does_not_leak() -> None:
    c = _app_with_checks(HealthRegistry(service="t"))
    r = c.get("/crash")
    assert r.status_code == 500
    assert r.json()["error"]["code"] == "internal_error"
    assert "secret internal detail" not in r.text


def test_framework_404_uses_envelope() -> None:
    c = _app_with_checks(HealthRegistry(service="t"))
    r = c.get("/does-not-exist", headers={REQUEST_ID_HEADER: "rid-404"})
    assert r.status_code == 404
    assert r.json()["error"] == {
        "code": "not_found",
        "message": "Not Found",
        "request_id": "rid-404",
    }


def test_request_id_visible_inside_handler() -> None:
    c = _app_with_checks(HealthRegistry(service="t"))
    r = c.get("/whoami", headers={REQUEST_ID_HEADER: "rid-2"})
    assert r.json() == {"request_id": "rid-2"}


@pytest.mark.parametrize("path", ["/health/live", "/health/ready"])
def test_health_endpoints_are_fast(client: TestClient, path: str) -> None:
    import time

    t = time.perf_counter()
    client.get(path)
    assert (time.perf_counter() - t) < 0.5
