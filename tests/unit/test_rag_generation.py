from __future__ import annotations

import json
from typing import Any

import httpx
import pytest

from llmops_core.config import Settings
from llmops_core.context import _request_id
from llmops_core.errors import UpstreamTimeoutError, UpstreamUnavailableError
from llmops_registry.prompts import PromptRegistry
from rag_service.generation import Generator, check_citations

SUPPLIED = ["FIN-001@v2#c3 (p.1)", "FIN-002@v1#c7 (p.4)"]


def test_citation_with_page_suffix_is_valid() -> None:
    check = check_citations("Revenue was EUR 4.2m [FIN-001@v2#c3 (p.1)].", SUPPLIED)
    assert check.cited == ["FIN-001@v2#c3"]
    assert check.valid


def test_citation_without_page_suffix_is_valid() -> None:
    assert check_citations("EUR 4.2m [FIN-001@v2#c3]", SUPPLIED).valid


def test_fabricated_citation_is_invalid() -> None:
    check = check_citations("EUR 4.2m [FIN-009@v1#c1]", SUPPLIED)
    assert check.invalid == ["FIN-009@v1#c1"]
    assert not check.valid


def test_superseded_version_is_not_the_same_source() -> None:
    # v1 of FIN-001 was not in the context; citing it is citing something unseen.
    assert not check_citations("[FIN-001@v1#c3]", SUPPLIED).valid


def test_grouped_citations_are_split_and_deduplicated() -> None:
    check = check_citations("a [FIN-001@v2#c3; FIN-002@v1#c7] b [FIN-001@v2#c3]", SUPPLIED)
    assert check.cited == ["FIN-001@v2#c3", "FIN-002@v1#c7"]


def test_answer_without_citations_has_none() -> None:
    check = check_citations("Revenue grew.", SUPPLIED)
    assert check.cited == [] and check.valid


def _generator(handler: Any, **kwargs: Any) -> Generator:
    prompt = PromptRegistry().get("rag", "rag_answer", "production")
    client = httpx.AsyncClient(base_url="http://api", transport=httpx.MockTransport(handler))
    return Generator(client, prompt, max_tokens=kwargs.get("max_tokens", 100), timeout_s=5.0)


def _reply(answer: str, model: str = "mock-1") -> httpx.Response:
    return httpx.Response(
        200, json={"answer": answer, "model": model, "input_tokens": 50, "output_tokens": 9}
    )


async def test_generator_sends_rendered_prompt_at_temperature_zero() -> None:
    seen: dict[str, Any] = {}

    def handler(req: httpx.Request) -> httpx.Response:
        seen["path"] = req.url.path
        seen["body"] = json.loads(req.content)
        seen["rid"] = req.headers.get("x-request-id")
        return _reply("EUR 4.2m [FIN-001@v2#c3]")

    token = _request_id.set("req-123")
    try:
        gen = await _generator(handler, max_tokens=77).generate(
            "What was revenue?", "<source id=...>", SUPPLIED
        )
    finally:
        _request_id.reset(token)

    assert seen["path"] == "/chat"
    content = seen["body"]["messages"][0]["content"]
    assert "What was revenue?" in content and "{question}" not in content
    assert seen["body"]["temperature"] == 0.0
    assert seen["body"]["max_tokens"] == 77
    assert seen["rid"] == "req-123"
    assert gen.prompt_ref == "rag/rag_answer@1.0.0"
    assert gen.citations.cited == ["FIN-001@v2#c3"]
    assert gen.input_tokens == 50 and not gen.abstained


@pytest.mark.parametrize("answer", ["insufficient_evidence", "`insufficient_evidence`."])
async def test_abstention_is_detected(answer: str) -> None:
    gen = await _generator(lambda _: _reply(answer)).generate("q", "ctx", SUPPLIED)
    assert gen.abstained


async def test_timeout_maps_to_504() -> None:
    def handler(req: httpx.Request) -> httpx.Response:
        raise httpx.ReadTimeout("slow", request=req)

    with pytest.raises(UpstreamTimeoutError):
        await _generator(handler).generate("q", "ctx", SUPPLIED)


async def test_unreachable_generator_maps_to_503() -> None:
    def handler(req: httpx.Request) -> httpx.Response:
        raise httpx.ConnectError("refused", request=req)

    with pytest.raises(UpstreamUnavailableError):
        await _generator(handler).generate("q", "ctx", SUPPLIED)


async def test_generator_5xx_maps_to_503() -> None:
    with pytest.raises(UpstreamUnavailableError):
        await _generator(lambda _: httpx.Response(502)).generate("q", "ctx", SUPPLIED)


def test_generation_is_off_unless_configured() -> None:
    assert Settings().rag.generator_url is None
