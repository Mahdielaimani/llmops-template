"""Grounded generation: rendered prompt -> llm-application /chat -> checked answer.

rag-service does not hold a model client. It calls llm-application over HTTP by its
Compose service name, so there is one egress point to models today and Phase 15's
model gateway replaces a URL, not a code path.

Citation validity is checked here, deterministically, before any judge sees the
answer. Whether `[X]` names a source that was actually in the context is a set
membership question; spending an LLM call on it would add cost and error to a
question that has an exact answer.
"""

from __future__ import annotations

import re
import time
from dataclasses import dataclass, field

import httpx

from llmops_core.context import REQUEST_ID_HEADER, get_request_id
from llmops_core.errors import UpstreamTimeoutError, UpstreamUnavailableError
from llmops_core.logging import get_logger
from llmops_registry.prompts import Prompt

log = get_logger("rag_service.generation")

ABSTAIN = "insufficient_evidence"
_BRACKET = re.compile(r"\[([^\[\]]+)\]")
_PAGE_SUFFIX = re.compile(r"\s*\(p\.\s*\d+\)\s*$")


@dataclass(frozen=True)
class CitationCheck:
    cited: list[str] = field(default_factory=list)
    invalid: list[str] = field(default_factory=list)

    @property
    def valid(self) -> bool:
        return not self.invalid


@dataclass(frozen=True)
class Generation:
    answer: str
    abstained: bool
    citations: CitationCheck
    prompt_ref: str
    model: str
    latency_ms: float
    input_tokens: int | None
    output_tokens: int | None


def _base_id(citation: str) -> str:
    return _PAGE_SUFFIX.sub("", citation).strip()


def check_citations(answer: str, supplied: list[str]) -> CitationCheck:
    """Every bracketed reference must name a supplied source. The page suffix is
    optional because models drop it often and it carries no identity — the chunk id
    already pins the page. Anything else in brackets that is not a supplied source is
    invalid, which is the point: a fabricated id must not pass as a citation."""
    allowed = {_base_id(c) for c in supplied}
    cited: list[str] = []
    invalid: list[str] = []
    for group in _BRACKET.findall(answer):
        for ref in (part.strip() for part in group.split(";")):
            if not ref:
                continue
            base = _base_id(ref)
            if base in allowed:
                if base not in cited:
                    cited.append(base)
            elif ref not in invalid:
                invalid.append(ref)
    return CitationCheck(cited=cited, invalid=invalid)


class Generator:
    def __init__(
        self,
        client: httpx.AsyncClient,
        prompt: Prompt,
        *,
        max_tokens: int,
        timeout_s: float,
    ) -> None:
        self.client = client
        self.prompt = prompt
        self.max_tokens = max_tokens
        self.timeout_s = timeout_s

    async def generate(self, question: str, context: str, supplied: list[str]) -> Generation:
        rendered = self.prompt.render(context=context, question=question)
        headers = {REQUEST_ID_HEADER: rid} if (rid := get_request_id()) else {}
        # temperature 0: evaluation compares runs, and sampling noise would be
        # indistinguishable from a prompt or retrieval change.
        payload = {
            "messages": [{"role": "user", "content": rendered}],
            "temperature": 0.0,
            "max_tokens": self.max_tokens,
        }
        started = time.perf_counter()
        try:
            resp = await self.client.post(
                "/chat", json=payload, headers=headers, timeout=self.timeout_s
            )
        except httpx.TimeoutException as exc:
            raise UpstreamTimeoutError(f"generation exceeded {self.timeout_s}s") from exc
        except httpx.TransportError as exc:
            raise UpstreamUnavailableError(f"llm-application unreachable: {exc}") from exc
        if resp.status_code >= 500:
            raise UpstreamUnavailableError(f"llm-application returned {resp.status_code}")
        resp.raise_for_status()
        body = resp.json()

        answer = body["answer"].strip()
        model = body["model"]
        if self.prompt.model_compatibility and model not in self.prompt.model_compatibility:
            # Not fatal: the compatibility list is a claim about what was evaluated,
            # and running outside it is how a new model gets evaluated at all.
            log.warning("prompt_model_unvalidated", prompt=self.prompt.ref, model=model)

        return Generation(
            answer=answer,
            abstained=answer.strip("`. ").lower() == ABSTAIN,
            citations=check_citations(answer, supplied),
            prompt_ref=self.prompt.ref,
            model=model,
            latency_ms=round((time.perf_counter() - started) * 1000, 3),
            input_tokens=body.get("input_tokens"),
            output_tokens=body.get("output_tokens"),
        )
