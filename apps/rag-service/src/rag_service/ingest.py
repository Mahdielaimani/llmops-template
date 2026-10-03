"""Ingestion: documents in, searchable chunks out.

    uv run python -m rag_service.ingest

    validate -> parse -> clean -> extract metadata -> chunk -> embed -> index

Every stage is separate and measured, because when retrieval quality is wrong the
question is always *which stage*. Chunking parameters are a versioned artifact
(docs/versioning.md class 5): changing them invalidates the index, so the
collection name carries a hash of the config.
"""

from __future__ import annotations

import hashlib
import json
import re
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Any

from qdrant_client import QdrantClient
from qdrant_client.models import Distance, PointStruct, VectorParams

from llmops_core.logging import get_logger
from rag_service.acl import Chunk, payload_of
from rag_service.embedding import Embedder

log = get_logger("ingest")

CORPUS = Path(__file__).resolve().parents[4] / "data" / "raw" / "corpus"


@dataclass(frozen=True)
class ChunkConfig:
    """Versioned: the index is only valid for the config that produced it."""

    target_tokens: int = 180
    overlap_tokens: int = 40
    # Financial text runs ~2.34 chars/token against ~5.70 for prose (measured in
    # docs/inference.md §1), so a character budget derived from prose would produce
    # chunks ~2.4x larger in tokens than intended.
    chars_per_token: float = 2.34

    @property
    def target_chars(self) -> int:
        return int(self.target_tokens * self.chars_per_token)

    @property
    def overlap_chars(self) -> int:
        return int(self.overlap_tokens * self.chars_per_token)

    def config_hash(self) -> str:
        raw = f"{self.target_tokens}:{self.overlap_tokens}:{self.chars_per_token}"
        return hashlib.sha256(raw.encode()).hexdigest()[:8]


def index_version(cfg: ChunkConfig, embed_model: str, corpus_sha: str) -> str:
    """ADR-015: the index identity is a hash of everything that produced it, so an
    index can never be confused about which embedder or chunking it belongs to."""
    raw = f"{cfg.config_hash()}|{embed_model}|{corpus_sha}"
    return hashlib.sha256(raw.encode()).hexdigest()[:12]


def clean(text: str) -> str:
    """Collapse whitespace, drop control characters. Deliberately conservative:
    aggressive cleaning destroys the number formatting the answers depend on."""
    text = text.replace(" ", " ")  # noqa: RUF001 - nbsp is the thing being removed
    text = "".join(ch for ch in text if ch == "\n" or ch >= " ")
    text = re.sub(r"[ \t]+", " ", text)
    return re.sub(r"\n{3,}", "\n\n", text).strip()


def split(text: str, cfg: ChunkConfig) -> list[str]:
    """Paragraph-aware with overlap. Splitting mid-sentence inside a financial
    figure ("EUR 1,234," | "567.89") would make the fact unretrievable, so
    paragraph boundaries are preferred and only oversized paragraphs are cut.
    """
    paragraphs = [p.strip() for p in text.split("\n\n") if p.strip()]
    chunks: list[str] = []
    current = ""

    for para in paragraphs:
        if len(para) > cfg.target_chars * 1.5:
            if current:
                chunks.append(current)
                current = ""
            sentences = re.split(r"(?<=[.!?])\s+", para)
            buf = ""
            for sentence in sentences:
                if buf and len(buf) + len(sentence) > cfg.target_chars:
                    chunks.append(buf.strip())
                    buf = buf[-cfg.overlap_chars :] + " " + sentence
                else:
                    buf = f"{buf} {sentence}".strip()
            if buf:
                current = buf
            continue

        if current and len(current) + len(para) > cfg.target_chars:
            chunks.append(current)
            current = current[-cfg.overlap_chars :] + "\n\n" + para
        else:
            current = f"{current}\n\n{para}".strip()

    if current:
        chunks.append(current)
    return [c.strip() for c in chunks if c.strip()]


def load_documents(directory: Path = CORPUS) -> list[dict[str, Any]]:
    if not directory.exists():
        raise FileNotFoundError(f"no corpus at {directory}; run `python -m rag_service.corpus`")
    return [json.loads(p.read_text(encoding="utf-8")) for p in sorted(directory.glob("*.json"))]


def corpus_sha(docs: list[dict[str, Any]]) -> str:
    """Content-addressed snapshot, so an index can be rebuilt byte-identically."""
    h = hashlib.sha256()
    for d in sorted(docs, key=lambda x: (x["doc_id"], x["version"])):
        h.update(f"{d['doc_id']}:{d['version']}:{''.join(d['pages'])}".encode())
    return h.hexdigest()[:12]


def to_chunks(docs: list[dict[str, Any]], cfg: ChunkConfig) -> list[Chunk]:
    out: list[Chunk] = []
    for d in docs:
        for page_no, raw_page in enumerate(d["pages"], start=1):
            for i, text in enumerate(split(clean(raw_page), cfg)):
                out.append(
                    Chunk(
                        chunk_id=f"{d['doc_id']}.v{d['version']}.p{page_no}.c{i}",
                        doc_id=d["doc_id"],
                        doc_version=d["version"],
                        title=d["title"],
                        text=text,
                        page=page_no,
                        classification=d["classification"],
                        department=d["department"],
                        effective_date=d["effective_date"],
                    )
                )
    return out


def run(
    client: QdrantClient,
    embedder: Embedder,
    cfg: ChunkConfig | None = None,
    directory: Path = CORPUS,
) -> dict[str, Any]:
    cfg = cfg or ChunkConfig()
    stages: dict[str, float] = {}

    t = time.perf_counter()
    docs = load_documents(directory)
    stages["load_ms"] = (time.perf_counter() - t) * 1000

    sha = corpus_sha(docs)
    version = index_version(cfg, embedder.model_name, sha)
    collection = f"docs_{version}"

    t = time.perf_counter()
    chunks = to_chunks(docs, cfg)
    stages["chunk_ms"] = (time.perf_counter() - t) * 1000

    t = time.perf_counter()
    vectors = embedder.embed_documents([c.text for c in chunks])
    stages["embed_ms"] = (time.perf_counter() - t) * 1000

    t = time.perf_counter()
    if client.collection_exists(collection):
        client.delete_collection(collection)
    client.create_collection(
        collection_name=collection,
        vectors_config=VectorParams(size=embedder.dim, distance=Distance.COSINE),
    )
    client.upsert(
        collection_name=collection,
        points=[
            PointStruct(id=i, vector=v.tolist(), payload=payload_of(c))
            for i, (c, v) in enumerate(zip(chunks, vectors, strict=True))
        ],
    )
    stages["index_ms"] = (time.perf_counter() - t) * 1000

    result = {
        "collection": collection,
        "index_version": version,
        "corpus_sha": sha,
        "chunk_config_hash": cfg.config_hash(),
        "embed_model": embedder.model_name,
        "embed_dim": embedder.dim,
        "documents": len(docs),
        "chunks": len(chunks),
        "chars_per_chunk_mean": round(sum(len(c.text) for c in chunks) / len(chunks), 1),
        **{k: round(v, 1) for k, v in stages.items()},
    }
    log.info("ingest_complete", **result)
    return result


if __name__ == "__main__":  # pragma: no cover - CLI
    import os

    url = os.getenv("LLMOPS_QDRANT__URL", "http://localhost:6333")
    summary = run(QdrantClient(url=url), Embedder())
    print(json.dumps(summary, indent=2))
