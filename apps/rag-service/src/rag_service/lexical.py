"""BM25 lexical index.

Exists because dense embeddings are bad at exact tokens. A document code like
`AUD-2026-014` compresses into a vector near every other code, while BM25 scores
it as a rare term and wins outright. The reverse holds for paraphrase, which is
why the two are fused rather than chosen between (ADR-003).

**The ACL applies here too.** A lexical index without the same predicate is a
bypass of the Qdrant filter — same documents, different door. The payloads
carry the identical fields and `AclPredicate.allows()` is the same callable
(ADR-019).

In-process because the corpus is 18 chunks. At real scale this is Elasticsearch
or OpenSearch, or Qdrant's own sparse vectors; see ADR-003 for when that flips.
"""

from __future__ import annotations

import re
import time
from dataclasses import dataclass
from typing import Any

from rank_bm25 import BM25Okapi

from llmops_core.logging import get_logger
from rag_service.acl import AclPredicate, Chunk, chunk_of

log = get_logger("lexical")

# Keep digits, letters and the internal hyphens/dots that make up identifiers:
# naive \w+ would split AUD-2026-014 into three common tokens and destroy the
# exact-match signal this index exists to provide.
_TOKEN = re.compile(r"[A-Za-z0-9]+(?:[-_./][A-Za-z0-9]+)*")


def tokenize(text: str) -> list[str]:
    out: list[str] = []
    for match in _TOKEN.findall(text.lower()):
        out.append(match)
        # Also index the parts, so "2026" and "014" match a query that only
        # mentions a fragment of the code.
        if any(sep in match for sep in "-_./"):
            out.extend(p for p in re.split(r"[-_./]", match) if p)
    return out


@dataclass
class LexicalResult:
    chunks: list[Chunk]
    search_ms: float
    candidates_scanned: int


class LexicalIndex:
    """Built once from the same payloads stored in Qdrant, so the two retrievers
    cannot drift apart on content or on ACL fields."""

    def __init__(self, payloads: list[dict[str, Any]]) -> None:
        self.payloads = payloads
        # The identifier is indexed alongside the text: a user searching for a
        # document code is searching metadata, not prose.
        corpus = [tokenize(f"{p['doc_id']} {p['title']} {p['text']}") for p in payloads]
        self.bm25 = BM25Okapi(corpus)
        log.info("lexical_index_built", chunks=len(payloads))

    def search(self, question: str, acl: AclPredicate, *, candidates: int = 20) -> LexicalResult:
        t = time.perf_counter()
        scores = self.bm25.get_scores(tokenize(question))

        # The ACL is applied while selecting candidates, not afterwards: an
        # unauthorised chunk must never enter the ranked list, exactly as in the
        # Qdrant filter.
        ranked = sorted(
            (
                (float(score), payload)
                for score, payload in zip(scores, self.payloads, strict=True)
                if score > 0 and acl.allows(payload)
            ),
            key=lambda pair: pair[0],
            reverse=True,
        )[:candidates]

        search_ms = (time.perf_counter() - t) * 1000
        return LexicalResult(
            chunks=[chunk_of(p, score=s, by="bm25") for s, p in ranked],
            search_ms=round(search_ms, 3),
            candidates_scanned=len(self.payloads),
        )


def load_payloads(client: Any, collection: str) -> list[dict[str, Any]]:
    """Read the payloads back out of Qdrant to build the lexical index.

    Deliberate: one ingestion writes one set of payloads, and both retrievers
    read from it. Building the BM25 index from the source documents instead would
    allow the two indexes to diverge after a partial re-ingest.
    """
    payloads: list[dict[str, Any]] = []
    offset = None
    while True:
        points, offset = client.scroll(
            collection_name=collection, limit=256, offset=offset, with_payload=True
        )
        payloads.extend(p.payload for p in points if p.payload)
        if offset is None:
            break
    return payloads
