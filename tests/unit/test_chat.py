from __future__ import annotations

import json

import httpx
import pytest
from fastapi.testclient import TestClient

from llm_application.providers import ChatRequest, Message, MockProvider
from llm_application.providers.openai_compat import OpenAICompatProvider
from llmops_core.context import REQUEST_ID_HEADER


def _body(**kw: object) -> dict[str, object]:
    return {"messages": [{"role": "user", "content": "what was Q2 revenue?"}], **kw}


def test_chat_returns_answer_and_telemetry(client: TestClient) -> None:
    r = client.post("/chat", json=_body())
    assert r.status_code == 200
    b = r.json()
    assert "what was Q2 revenue?" in b["answer"]
    assert b["model"] == "mock-1"
    assert b["finish_reason"] == "stop"
    assert b["request_id"] == r.headers[REQUEST_ID_HEADER]
    assert b["latency_ms"] >= 0
    assert b["output_tokens"] > 0
    assert b["total_tokens"] == b["input_tokens"] + b["output_tokens"]
    assert b["grounded"] is False


def test_max_tokens_truncates(client: TestClient) -> None:
    r = client.post("/chat", json=_body(max_tokens=3))
    assert r.json()["output_tokens"] == 3


def test_validation_rejects_empty_messages(client: TestClient) -> None:
    r = client.post("/chat", json={"messages": []})
    assert r.status_code == 422
    assert r.json()["error"]["code"] == "validation_error"


def test_validation_rejects_bad_role(client: TestClient) -> None:
    r = client.post("/chat", json={"messages": [{"role": "root", "content": "x"}]})
    assert r.status_code == 422


def test_validation_rejects_out_of_range_temperature(client: TestClient) -> None:
    assert client.post("/chat", json=_body(temperature=5)).status_code == 422


def _events(text: str) -> list[tuple[str, dict]]:
    out = []
    for block in text.strip().split("\n\n"):
        lines = dict(line.split(": ", 1) for line in block.splitlines())
        out.append((lines["event"], json.loads(lines["data"])))
    return out


def test_stream_emits_deltas_then_done(client: TestClient) -> None:
    r = client.post("/chat", json=_body(stream=True))
    assert r.status_code == 200
    assert r.headers["content-type"].startswith("text/event-stream")
    assert r.headers["x-accel-buffering"] == "no"

    events = _events(r.text)
    names = [n for n, _ in events]
    assert names[-1] == "done"
    assert names.count("delta") > 1

    final = events[-1][1]
    reassembled = "".join(d["delta"] for n, d in events if n == "delta")
    assert reassembled == final["answer"]
    assert final["ttft_ms"] is not None


def test_stream_preserves_request_id_header(client: TestClient) -> None:
    """The middleware is pure ASGI precisely so streaming keeps working — prove it."""
    r = client.post("/chat", json=_body(stream=True), headers={REQUEST_ID_HEADER: "stream-1"})
    assert r.headers[REQUEST_ID_HEADER] == "stream-1"
    assert _events(r.text)[-1][1]["request_id"] == "stream-1"


async def test_mock_timings_separate_prefill_from_decode() -> None:
    p = MockProvider(prefill_s=0.05, per_token_s=0.01)
    req = ChatRequest(messages=[Message(role="user", content="a b c d e")], max_tokens=5)
    result = await p.complete(req)
    assert result.timings.ttft_ms is not None
    # asyncio.sleep(0.05) can return a hair under 50 ms on Windows clock resolution,
    # so assert the prefill delay is reflected rather than an exact floor.
    assert result.timings.ttft_ms >= 45
    assert result.timings.total_ms > result.timings.ttft_ms
    assert result.timings.tpot_ms is not None
    assert result.timings.tokens_per_second is not None


async def test_usage_is_none_when_provider_omits_it() -> None:
    """A guessed token count would silently corrupt the cost model."""
    calls: list[dict] = []

    async def handler(request: httpx.Request) -> httpx.Response:
        calls.append(json.loads(request.content))
        return httpx.Response(
            200,
            json={
                "model": "served-model",
                "choices": [{"message": {"content": "hi"}, "finish_reason": "stop"}],
            },
        )

    p = OpenAICompatProvider(base_url="http://vllm:8000/v1", api_key=None, model="m", timeout_s=5)
    p._client = httpx.AsyncClient(
        transport=httpx.MockTransport(handler), base_url="http://vllm:8000/v1"
    )
    result = await p.complete(ChatRequest(messages=[Message(role="user", content="hi")]))
    assert result.usage.input_tokens is None
    assert result.usage.total is None
    assert result.model == "served-model"
    assert calls[0]["stream"] is False
    await p.aclose()


@pytest.mark.parametrize(
    ("status", "code"),
    [(429, "rate_limited"), (500, "upstream_unavailable"), (504, "upstream_timeout")],
)
async def test_upstream_status_maps_to_envelope(status: int, code: str) -> None:
    async def handler(request: httpx.Request) -> httpx.Response:
        return httpx.Response(status, text="upstream said something with the prompt in it")

    p = OpenAICompatProvider(base_url="http://x/v1", api_key=None, model="m", timeout_s=5)
    p._client = httpx.AsyncClient(transport=httpx.MockTransport(handler), base_url="http://x/v1")
    with pytest.raises(Exception) as exc:
        await p.complete(ChatRequest(messages=[Message(role="user", content="hi")]))
    assert exc.value.code == code  # type: ignore[attr-defined]
    await p.aclose()


async def test_stream_ignores_role_only_first_chunk() -> None:
    """An OpenAI-compatible server opens with a role delta carrying no content; timing
    TTFT from it would report a first token that was never generated."""
    chunks = [
        '{"model":"m","choices":[{"delta":{"role":"assistant"}}]}',
        '{"model":"m","choices":[{"delta":{"content":"He"}}]}',
        '{"model":"m","choices":[{"delta":{"content":"llo"},"finish_reason":"stop"}]}',
        '{"model":"m","choices":[],"usage":{"prompt_tokens":7,"completion_tokens":2}}',
    ]
    body = "".join(f"data: {c}\n\n" for c in chunks) + "data: [DONE]\n\n"

    def handler(request: httpx.Request) -> httpx.Response:
        return httpx.Response(200, text=body, headers={"content-type": "text/event-stream"})

    p = OpenAICompatProvider(base_url="http://x/v1", api_key=None, model="m", timeout_s=5)
    p._client = httpx.AsyncClient(transport=httpx.MockTransport(handler), base_url="http://x/v1")

    deltas, final = [], None
    async for chunk in p.stream(ChatRequest(messages=[Message(role="user", content="hi")])):
        if chunk.done:
            final = chunk.result
        else:
            deltas.append(chunk.delta)

    assert deltas == ["He", "llo"]
    assert final is not None
    assert final.content == "Hello"
    assert final.finish_reason == "stop"
    assert final.usage.input_tokens == 7
    assert final.usage.output_tokens == 2
    assert len(final.timings.inter_token_ms) == 1  # 2 tokens => 1 gap
    await p.aclose()
