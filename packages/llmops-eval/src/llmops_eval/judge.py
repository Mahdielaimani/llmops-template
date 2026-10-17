"""LLM-as-a-Judge: rubric prompts from the registry, verdicts validated by schema.

The judge is a measuring instrument with its own error, so three rules:

- **Categorical verdicts, not 1-10 scores.** A small model cannot hold a ten-point
  scale steady across items; three labels with written definitions it can.
- **A verdict that fails to parse is a recorded outcome, never a dropped row.**
  Dropping failures biases the mean towards the items the judge found easy.
- **Arithmetic is recomputed, not trusted.** The faithfulness prompt asks for a
  fraction of supported claims; the score used is computed from the claims the
  judge listed, and a disagreement with its own number is logged.

The judge calls its model directly (OpenAI-compatible), not through
llm-application: it is evaluation infrastructure, and sharing the serving path
would let a judge outage look like a product incident and let the judge default
to the generator's model.
"""

from __future__ import annotations

import json
import re
import time
from dataclasses import asdict, dataclass, field
from typing import Any, Literal

import httpx
from pydantic import BaseModel, ValidationError

from llmops_registry.prompts import Prompt, PromptRegistry

Status = Literal["ok", "parse_error", "schema_error", "transport_error"]


class CorrectnessVerdict(BaseModel):
    reason: str
    verdict: Literal["correct", "partially_correct", "incorrect"]


class RelevanceVerdict(BaseModel):
    reason: str
    verdict: Literal["relevant", "partially_relevant", "irrelevant"]


class CompletenessVerdict(BaseModel):
    reason: str
    verdict: Literal["complete", "partial", "missing"]


class Claim(BaseModel):
    claim: str
    verdict: Literal["supported", "unsupported", "contradicted"]
    source_id: str | None = None


class FaithfulnessVerdict(BaseModel):
    claims: list[Claim]
    faithfulness: float | None = None


@dataclass(frozen=True)
class Rubric:
    name: str
    schema: type[BaseModel]
    scores: dict[str, float]
    needs_reference: bool = False


RUBRICS: dict[str, Rubric] = {
    "correctness": Rubric(
        "correctness",
        CorrectnessVerdict,
        {"correct": 1.0, "partially_correct": 0.5, "incorrect": 0.0},
        needs_reference=True,
    ),
    "relevance": Rubric(
        "relevance",
        RelevanceVerdict,
        {"relevant": 1.0, "partially_relevant": 0.5, "irrelevant": 0.0},
    ),
    "completeness": Rubric(
        "completeness", CompletenessVerdict, {"complete": 1.0, "partial": 0.5, "missing": 0.0}
    ),
    # Scored from the claim list, see _score.
    "faithfulness": Rubric("faithfulness", FaithfulnessVerdict, {}),
}


@dataclass
class JudgeResult:
    rubric: str
    status: Status
    prompt_ref: str
    judge_model: str
    latency_ms: float
    verdict: str | None = None
    score: float | None = None
    reason: str | None = None
    claims: list[dict[str, Any]] = field(default_factory=list)
    reported_score: float | None = None
    error: str | None = None
    raw: str | None = None

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


_FENCE = re.compile(r"^```(?:json)?\s*|\s*```$", re.MULTILINE)


def extract_json(text: str) -> Any:
    """Small models wrap JSON in fences or prose despite being told not to. Take
    the outermost object; anything beyond that is a parse error, not a guess."""
    cleaned = _FENCE.sub("", text.strip())
    start, end = cleaned.find("{"), cleaned.rfind("}")
    if start == -1 or end <= start:
        raise ValueError("no JSON object in judge output")
    return json.loads(cleaned[start : end + 1])


def _score(rubric: Rubric, parsed: BaseModel) -> tuple[str | None, float | None, float | None]:
    if isinstance(parsed, FaithfulnessVerdict):
        if not parsed.claims:
            # No claims: an abstention or an empty answer. Nothing is unsupported,
            # but scoring it 1.0 would reward saying nothing.
            return "no_claims", None, parsed.faithfulness
        supported = sum(c.verdict == "supported" for c in parsed.claims)
        return None, round(supported / len(parsed.claims), 4), parsed.faithfulness
    verdict = str(getattr(parsed, "verdict"))  # noqa: B009 — every non-faithfulness schema has it
    return verdict, rubric.scores[verdict], None


def interpret(rubric: Rubric, raw: str) -> tuple[Status, BaseModel | None, str | None]:
    try:
        payload = extract_json(raw)
    except (ValueError, json.JSONDecodeError) as exc:
        return "parse_error", None, str(exc)
    try:
        return "ok", rubric.schema.model_validate(payload), None
    except ValidationError as exc:
        return "schema_error", None, exc.errors()[0]["msg"]


class JudgeClient:
    def __init__(
        self,
        client: httpx.AsyncClient,
        model: str,
        *,
        registry: PromptRegistry | None = None,
        prompt_ref: str = "dev",
        timeout_s: float = 120.0,
    ) -> None:
        self.client = client
        self.model = model
        self.timeout_s = timeout_s
        registry = registry or PromptRegistry()
        self.prompts: dict[str, Prompt] = {
            name: registry.get("judge", name, prompt_ref) for name in RUBRICS
        }

    async def judge(self, rubric_name: str, **values: str) -> JudgeResult:
        rubric = RUBRICS[rubric_name]
        prompt = self.prompts[rubric_name]
        result = JudgeResult(
            rubric=rubric_name,
            status="ok",
            prompt_ref=prompt.ref,
            judge_model=self.model,
            latency_ms=0.0,
        )
        rendered = prompt.render(**{k: v for k, v in values.items() if k in prompt.variables})
        started = time.perf_counter()
        try:
            resp = await self.client.post(
                "/chat/completions",
                json={
                    "model": self.model,
                    "messages": [{"role": "user", "content": rendered}],
                    "temperature": 0.0,
                    "response_format": {"type": "json_object"},
                },
                timeout=self.timeout_s,
            )
            resp.raise_for_status()
            raw = resp.json()["choices"][0]["message"]["content"] or ""
        except (httpx.HTTPError, KeyError, IndexError) as exc:
            result.status = "transport_error"
            result.error = f"{type(exc).__name__}: {exc}"
            return result
        finally:
            result.latency_ms = round((time.perf_counter() - started) * 1000, 1)

        status, parsed, error = interpret(rubric, raw)
        result.status = status
        if parsed is None:
            result.error, result.raw = error, raw[:2000]
            return result

        result.verdict, result.score, result.reported_score = _score(rubric, parsed)
        result.reason = getattr(parsed, "reason", None)
        if isinstance(parsed, FaithfulnessVerdict):
            result.claims = [c.model_dump() for c in parsed.claims]
        return result
