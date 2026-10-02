"""Measure /predict against /chat on the same machine, same moment.

    docker compose --profile core --profile classical up -d --build
    uv run python benchmarks/compare_serving.py                    # from the host
    docker compose exec api python /build/benchmarks/compare_serving.py --inside

Reports two latencies per endpoint, because on this machine they differ by two
orders of magnitude:

  server_ms      — measured inside the handler, reported in the response body
  roundtrip_ms   — measured by this client

From the Windows host, roundtrip is dominated by Docker Desktop's WSL2 port proxy
(~43 ms, constant, identical for every endpoint) and tells you nothing about the
service. Run with --inside to measure over the Compose network instead.

The LLM column uses the mock provider: framework overhead with zero inference
cost, so it is a floor and not a comparison. Phase 6 replaces it with vLLM.
"""

from __future__ import annotations

import json
import statistics
import sys
import time

import httpx

INSIDE = "--inside" in sys.argv
CLASSICAL = "http://classical-ml:8090" if INSIDE else "http://localhost:8090"
LLM = "http://api:8080" if INSIDE else "http://localhost:8080"
N = 200

INVOICE = {
    "amount_eur": 24500.0,
    "days_to_due": 30,
    "customer_prior_invoices": 12,
    "customer_prior_late_ratio": 0.25,
    "has_purchase_order": True,
    "is_new_customer": False,
}
CHAT = {
    "messages": [{"role": "user", "content": "will invoice 4471 be paid late?"}],
    "max_tokens": 32,
}


def percentiles(samples: list[float]) -> dict[str, float]:
    s = sorted(samples)
    return {
        "p50": round(statistics.median(s), 3),
        "p95": round(s[int(len(s) * 0.95)], 3),
        "p99": round(s[int(len(s) * 0.99)], 3),
        "mean": round(statistics.fmean(s), 3),
    }


def _server_ms(payload: dict) -> float | None:
    """The handler's own measurement, so client-side transport can be subtracted."""
    for key in ("latency_ms", "total_ms"):
        if isinstance(payload.get(key), int | float):
            return float(payload[key])
    return None


def bench(client: httpx.Client, url: str, body: dict, n: int) -> dict[str, object]:
    client.post(url, json=body)  # warm the path, exclude first-call cost
    roundtrip, server = [], []
    started = time.perf_counter()
    for _ in range(n):
        t = time.perf_counter()
        r = client.post(url, json=body)
        r.raise_for_status()
        roundtrip.append((time.perf_counter() - t) * 1000)
        ms = _server_ms(r.json())
        if ms is not None:
            server.append(ms)
    wall = time.perf_counter() - started
    out: dict[str, object] = {
        "roundtrip_ms": percentiles(roundtrip),
        "rps_sequential": round(n / wall, 1),
        "response_bytes": len(r.content),
    }
    if server:
        out["server_ms"] = percentiles(server)
        out["transport_overhead_ms"] = round(
            percentiles(roundtrip)["p50"] - percentiles(server)["p50"], 3
        )
    return out


def main() -> int:
    with httpx.Client(timeout=30.0) as c:
        try:
            card = c.get(f"{CLASSICAL}/model").json()
        except httpx.ConnectError:
            print(f"classical-ml not reachable at {CLASSICAL}", file=sys.stderr)
            return 1
        out = {
            "measured_from": "compose network" if INSIDE else "windows host",
            "model_card": {k: card[k] for k in ("name", "version", "artifact_bytes", "metrics")},
            "classical_single": bench(c, f"{CLASSICAL}/predict", INVOICE, N),
            "classical_batch_100": bench(
                c, f"{CLASSICAL}/predict/batch", {"invoices": [INVOICE] * 100}, 20
            ),
            "llm_mock": bench(c, f"{LLM}/chat", CHAT, N),
        }
    print(json.dumps(out, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
