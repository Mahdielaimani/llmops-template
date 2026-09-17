"""Liveness / readiness primitives.

Liveness  (``/health/live``)  — "is the process alive?"  Never checks
dependencies; a failing dependency must not get the pod restarted.

Readiness (``/health/ready``) — "can this instance take traffic *now*?"
Runs registered dependency checks (Postgres, Redis, Qdrant, model server…)
with a short timeout each. Failing readiness pulls the instance out of the
load balancer / K8s Service without killing it.

Startup   (K8s ``startupProbe``) — reuses readiness with a long
``failureThreshold``; matters for LLM servers whose model load takes minutes.

Checks are registered by each service at startup; the core only provides the
registry, the aggregation, and the HTTP router.
"""

from __future__ import annotations

import asyncio
import time
from collections.abc import Awaitable, Callable
from dataclasses import dataclass, field
from typing import Literal

from fastapi import APIRouter, Response, status
from pydantic import BaseModel

from llmops_core.version import __version__

CheckFn = Callable[[], Awaitable[None]]
Status = Literal["ok", "fail", "timeout"]


class CheckResult(BaseModel):
    name: str
    status: Status
    latency_ms: float
    error: str | None = None


class HealthResponse(BaseModel):
    status: Literal["ok", "degraded"]
    service: str
    version: str
    checks: list[CheckResult] = []


@dataclass
class HealthRegistry:
    service: str
    default_timeout_s: float = 2.0
    _checks: dict[str, tuple[CheckFn, float]] = field(default_factory=dict)

    def register(self, name: str, fn: CheckFn, *, timeout_s: float | None = None) -> None:
        self._checks[name] = (fn, timeout_s or self.default_timeout_s)

    async def _run_one(self, name: str, fn: CheckFn, timeout_s: float) -> CheckResult:
        started = time.perf_counter()
        try:
            await asyncio.wait_for(fn(), timeout=timeout_s)
            st: Status = "ok"
            err: str | None = None
        except TimeoutError:
            st, err = "timeout", f"exceeded {timeout_s}s"
        except Exception as exc:  # any failure = not ready; message is enough
            st, err = "fail", f"{type(exc).__name__}: {exc}"
        return CheckResult(
            name=name,
            status=st,
            latency_ms=round((time.perf_counter() - started) * 1000, 2),
            error=err,
        )

    async def readiness(self) -> HealthResponse:
        results = await asyncio.gather(
            *(self._run_one(n, fn, t) for n, (fn, t) in self._checks.items())
        )
        overall: Literal["ok", "degraded"] = (
            "ok" if all(r.status == "ok" for r in results) else "degraded"
        )
        return HealthResponse(
            status=overall, service=self.service, version=__version__, checks=list(results)
        )

    def liveness(self) -> HealthResponse:
        return HealthResponse(status="ok", service=self.service, version=__version__)


def build_health_router(registry: HealthRegistry) -> APIRouter:
    router = APIRouter(prefix="/health", tags=["health"])

    @router.get("/live", response_model=HealthResponse)
    async def live() -> HealthResponse:
        return registry.liveness()

    @router.get("/ready", response_model=HealthResponse)
    async def ready(response: Response) -> HealthResponse:
        result = await registry.readiness()
        if result.status != "ok":
            response.status_code = status.HTTP_503_SERVICE_UNAVAILABLE
        return result

    return router
