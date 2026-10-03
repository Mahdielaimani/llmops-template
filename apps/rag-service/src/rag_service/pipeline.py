"""The four retrieval configurations, behind one call.

    dense          vector only
    lexical        BM25 only
    hybrid         RRF(dense, lexical)
    hybrid+rerank  RRF then cross-encoder

They exist as one enum rather than four code paths so Phase 11 can evaluate them
against the same questions with the same ACL and the only difference being the
mode. A comparison where the configurations are separate implementations
measures the implementations, not the strategies.
"""

from __future__ import annotations

import time
from dataclasses import dataclass, field
from enum import StrEnum

from llmops_core.logging import get_logger
from rag_service.acl import AclPredicate, Chunk
from rag_service.fusion import Reranker, reciprocal_rank_fusion, weighted_fusion
from rag_service.lexical import LexicalIndex
from rag_service.retriever import Retriever

log = get_logger("pipeline")


class Mode(StrEnum):
    DENSE = "dense"
    LEXICAL = "lexical"
    HYBRID = "hybrid"
    HYBRID_WEIGHTED = "hybrid_weighted"
    HYBRID_RERANK = "hybrid_rerank"


@dataclass
class PipelineResult:
    chunks: list[Chunk]
    mode: str
    acl_scope: str
    dense_candidates: int = 0
    lexical_candidates: int = 0
    fused_candidates: int = 0
    reranked: int = 0
    stages: dict[str, float] = field(default_factory=dict)

    @property
    def total_ms(self) -> float:
        return round(sum(self.stages.values()), 3)


class RetrievalPipeline:
    def __init__(
        self,
        retriever: Retriever,
        lexical: LexicalIndex,
        reranker: Reranker | None = None,
        *,
        dense_weight: float = 0.6,
    ) -> None:
        self.retriever = retriever
        self.lexical = lexical
        self.reranker = reranker
        self.dense_weight = dense_weight

    def run(
        self,
        question: str,
        acl: AclPredicate,
        *,
        mode: Mode = Mode.HYBRID_WEIGHTED,
        top_k: int = 5,
        candidates: int = 20,
    ) -> PipelineResult:
        stages: dict[str, float] = {}
        dense: list[Chunk] = []
        lexical: list[Chunk] = []

        if mode is not Mode.LEXICAL:
            dense_result = self.retriever.search(
                question, acl, top_k=candidates, candidates=candidates
            )
            dense = dense_result.chunks
            stages["embed_ms"] = dense_result.embed_ms
            stages["dense_ms"] = dense_result.search_ms
            stages["acl_recheck_ms"] = dense_result.recheck_ms

        if mode is not Mode.DENSE:
            lexical_result = self.lexical.search(question, acl, candidates=candidates)
            lexical = lexical_result.chunks
            stages["lexical_ms"] = lexical_result.search_ms

        t = time.perf_counter()
        if mode is Mode.DENSE:
            fused = dense
        elif mode is Mode.LEXICAL:
            fused = lexical
        elif mode is Mode.HYBRID_WEIGHTED:
            fused = weighted_fusion(dense, lexical, dense_weight=self.dense_weight)
        else:
            fused = reciprocal_rank_fusion(dense, lexical)
        stages["fusion_ms"] = round((time.perf_counter() - t) * 1000, 3)

        reranked = 0
        if mode is Mode.HYBRID_RERANK:
            if self.reranker is None:
                raise RuntimeError("hybrid_rerank requested but no reranker is configured")
            rr = self.reranker.rerank(question, fused[:candidates], top_k=top_k)
            stages["rerank_ms"] = rr.rerank_ms
            reranked = rr.reranked
            final = rr.chunks
        else:
            final = fused[:top_k]

        result = PipelineResult(
            chunks=final,
            mode=str(mode),
            acl_scope=acl.scope_hash(),
            dense_candidates=len(dense),
            lexical_candidates=len(lexical),
            fused_candidates=len(fused),
            reranked=reranked,
            stages=stages,
        )
        log.info(
            "pipeline",
            mode=str(mode),
            dense=len(dense),
            lexical=len(lexical),
            fused=len(fused),
            returned=len(final),
            total_ms=result.total_ms,
            **{k: v for k, v in stages.items()},
        )
        return result
