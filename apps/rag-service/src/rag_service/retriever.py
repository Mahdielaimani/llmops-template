"""Retrieval with the ACL applied as a query predicate.

The whole point of this module, from ADR-019: `acl.qdrant_filter()` goes **into**
the search call. A disallowed chunk is never a candidate, so there is nothing to
post-filter and nothing to leak through absence or through a collapsed top-k.

The re-check in `search()` is deliberately redundant. Two independent enforcement
points are the design: a bug in the filter and a bug in the re-check are unlikely
to coincide, and the re-check is the one that would catch a payload written
without the `classification_level` field.
"""

from __future__ import annotations

import time
from dataclasses import dataclass, field
from typing import Any

from qdrant_client import QdrantClient
from qdrant_client.http.models import Filter

from llmops_core.logging import get_logger
from rag_service.acl import AclPredicate, Chunk, chunk_of
from rag_service.embedding import Embedder

log = get_logger("retriever")


@dataclass
class RetrievalResult:
    chunks: list[Chunk]
    embed_ms: float
    search_ms: float
    recheck_ms: float
    candidates_returned: int
    rejected_by_recheck: int = 0
    acl_scope: str = ""
    collection: str = ""
    timings: dict[str, float] = field(default_factory=dict)

    @property
    def total_ms(self) -> float:
        return round(self.embed_ms + self.search_ms + self.recheck_ms, 3)


class Retriever:
    def __init__(self, client: QdrantClient, embedder: Embedder, collection: str) -> None:
        self.client = client
        self.embedder = embedder
        self.collection = collection

    def search(
        self, question: str, acl: AclPredicate, *, top_k: int = 5, candidates: int = 20
    ) -> RetrievalResult:
        t = time.perf_counter()
        vector = self.embedder.embed_query(question)
        embed_ms = (time.perf_counter() - t) * 1000

        t = time.perf_counter()
        hits = self.client.query_points(
            collection_name=self.collection,
            query=vector.tolist(),
            # The ACL predicate is part of the query, not applied to its output.
            query_filter=Filter(**acl.qdrant_filter()),
            limit=candidates,
            with_payload=True,
        ).points
        search_ms = (time.perf_counter() - t) * 1000

        t = time.perf_counter()
        allowed: list[Chunk] = []
        rejected = 0
        for h in hits:
            payload: dict[str, Any] = h.payload or {}
            if not acl.allows(payload):
                # Reaching here means the Qdrant filter and the predicate disagree,
                # which is a bug worth an explicit log rather than a silent drop.
                rejected += 1
                log.warning(
                    "acl_recheck_rejected",
                    chunk_id=payload.get("chunk_id"),
                    classification=payload.get("classification"),
                    acl_scope=acl.scope_hash(),
                )
                continue
            allowed.append(chunk_of(payload, score=float(h.score)))
        recheck_ms = (time.perf_counter() - t) * 1000

        result = RetrievalResult(
            chunks=allowed[:top_k],
            embed_ms=round(embed_ms, 3),
            search_ms=round(search_ms, 3),
            recheck_ms=round(recheck_ms, 3),
            candidates_returned=len(hits),
            rejected_by_recheck=rejected,
            acl_scope=acl.scope_hash(),
            collection=self.collection,
        )
        log.info(
            "retrieval",
            question_chars=len(question),
            candidates=len(hits),
            returned=len(result.chunks),
            rejected_by_recheck=rejected,
            acl_scope=result.acl_scope,
            embed_ms=result.embed_ms,
            search_ms=result.search_ms,
            recheck_ms=result.recheck_ms,
        )
        return result


MAX_CONTEXT_CHARS = 6000


def build_context(chunks: list[Chunk], max_chars: int = MAX_CONTEXT_CHARS) -> tuple[str, list[str]]:
    """Render chunks for the prompt, and return the citation list separately.

    Two deliberate choices. The chunks are framed as **data**, inside delimiters,
    never merged into the instruction — retrieved content is untrusted and may
    contain injected instructions (docs/security.md §2). And the citation list is
    returned alongside rather than parsed back out of the model's answer, so the
    output guardrail can check claims against what was actually supplied.
    """
    parts: list[str] = []
    citations: list[str] = []
    used = 0
    for c in chunks:
        block = (
            f'<source id="{c.citation}" title="{c.title}" '
            f'classification="{c.classification}" effective="{c.effective_date}">\n'
            f"{c.text}\n</source>"
        )
        if used + len(block) > max_chars:
            break
        parts.append(block)
        citations.append(c.citation)
        used += len(block)
    return "\n\n".join(parts), citations
