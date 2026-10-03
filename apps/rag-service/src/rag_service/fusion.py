"""Fusion and reranking.

Two stages that are often confused:

**Fusion** merges two ranked lists. Reciprocal Rank Fusion is the default because
it uses only *ranks*, so it needs no score calibration — a cosine similarity of
0.89 and a BM25 score of 7.3 are not comparable quantities, and any weighted sum
of them requires normalisation that depends on the query (ADR-003).

**Reranking** re-scores the fused candidates with a cross-encoder that reads the
query and the passage *together*. A bi-encoder embeds them separately and can
only measure vector proximity; a cross-encoder can represent "this passage
answers this question" (ADR-004). It is accurate and slow, which is why it runs
on ~20 candidates and never on the corpus.
"""

from __future__ import annotations

import time
from dataclasses import dataclass

from fastembed.rerank.cross_encoder import TextCrossEncoder

from llmops_core.logging import get_logger
from rag_service.acl import Chunk

log = get_logger("fusion")

RRF_K = 60  # the constant from the original RRF paper; damps the top-rank advantage
DEFAULT_RERANKER = "Xenova/ms-marco-MiniLM-L-6-v2"


def reciprocal_rank_fusion(
    dense: list[Chunk], lexical: list[Chunk], *, k: int = RRF_K
) -> list[Chunk]:
    """score(d) = sum over lists of 1 / (k + rank(d))

    Rank-based, so no score normalisation is needed. A chunk found by both
    retrievers outranks one found strongly by a single retriever, which is the
    behaviour that makes hybrid better than either part.
    """
    scores: dict[str, float] = {}
    best: dict[str, Chunk] = {}
    sources: dict[str, set[str]] = {}

    for ranked in (dense, lexical):
        for rank, chunk in enumerate(ranked, start=1):
            key = chunk.chunk_id
            scores[key] = scores.get(key, 0.0) + 1.0 / (k + rank)
            sources.setdefault(key, set()).add(chunk.retrieved_by)
            # Keep the first copy seen; the score is replaced by the fused one.
            best.setdefault(key, chunk)

    out: list[Chunk] = []
    for key, score in sorted(scores.items(), key=lambda kv: kv[1], reverse=True):
        chunk = best[key].model_copy(
            update={"score": round(score, 6), "retrieved_by": "+".join(sorted(sources[key]))}
        )
        out.append(chunk)
    return out


def weighted_fusion(
    dense: list[Chunk], lexical: list[Chunk], *, dense_weight: float = 0.6
) -> list[Chunk]:
    """Min-max normalised weighted sum. Offered for comparison, not as the default.

    The weakness is visible in the normalisation: if one retriever returns a
    single result, min-max maps it to 1.0 regardless of how good it is. RRF has
    no such failure mode because it never looks at the scores.
    """

    def normalise(chunks: list[Chunk]) -> dict[str, float]:
        if not chunks:
            return {}
        values = [c.score for c in chunks]
        lo, hi = min(values), max(values)
        span = hi - lo
        return {c.chunk_id: (1.0 if span == 0 else (c.score - lo) / span) for c in chunks}

    dn, ln = normalise(dense), normalise(lexical)
    best = {c.chunk_id: c for c in lexical} | {c.chunk_id: c for c in dense}
    combined = {
        key: dense_weight * dn.get(key, 0.0) + (1 - dense_weight) * ln.get(key, 0.0)
        for key in set(dn) | set(ln)
    }
    return [
        best[key].model_copy(update={"score": round(score, 6), "retrieved_by": "weighted"})
        for key, score in sorted(combined.items(), key=lambda kv: kv[1], reverse=True)
    ]


@dataclass
class RerankResult:
    chunks: list[Chunk]
    rerank_ms: float
    reranked: int


class Reranker:
    """Cross-encoder over the fused candidates.

    ONNX via fastembed, consistent with ADR-020: `ms-marco-MiniLM-L-6-v2` is
    80 MB against 1.04 GB for `bge-reranker-base`, and disk is the constraint.
    """

    def __init__(self, model_name: str = DEFAULT_RERANKER) -> None:
        self.model_name = model_name
        self._model = TextCrossEncoder(model_name=model_name)
        log.info("reranker_ready", model=model_name)

    def rerank(self, question: str, chunks: list[Chunk], *, top_k: int = 5) -> RerankResult:
        if not chunks:
            return RerankResult(chunks=[], rerank_ms=0.0, reranked=0)

        t = time.perf_counter()
        scores = list(self._model.rerank(question, [c.text for c in chunks]))
        rerank_ms = (time.perf_counter() - t) * 1000

        ordered = sorted(
            (
                c.model_copy(update={"score": round(float(s), 6), "retrieved_by": "rerank"})
                for c, s in zip(chunks, scores, strict=True)
            ),
            key=lambda c: c.score,
            reverse=True,
        )
        log.info(
            "rerank",
            model=self.model_name,
            candidates=len(chunks),
            returned=min(top_k, len(ordered)),
            rerank_ms=round(rerank_ms, 3),
        )
        return RerankResult(
            chunks=ordered[:top_k], rerank_ms=round(rerank_ms, 3), reranked=len(chunks)
        )
