from __future__ import annotations

import os
import time
from collections.abc import AsyncIterator
from contextlib import asynccontextmanager
from datetime import date

from fastapi import APIRouter, FastAPI, Header, Request
from pydantic import BaseModel
from qdrant_client import QdrantClient

from llmops_core import __version__
from llmops_core.config import Settings, get_settings
from llmops_core.context import RequestContextMiddleware, get_request_id
from llmops_core.errors import NotFoundError, install_error_handlers
from llmops_core.health import HealthRegistry, build_health_router
from llmops_core.logging import configure_logging, get_logger
from rag_service.acl import AclPredicate, Classification, RagRequest, User
from rag_service.embedding import Embedder
from rag_service.fusion import Reranker
from rag_service.lexical import LexicalIndex, load_payloads
from rag_service.pipeline import Mode, RetrievalPipeline
from rag_service.retriever import Retriever, build_context

log = get_logger("rag_service")
router = APIRouter(tags=["rag"])
SERVICE_NAME = "rag-service"


class Citation(BaseModel):
    citation: str
    doc_id: str
    doc_version: int
    title: str
    page: int
    classification: str
    score: float


class RagResponse(BaseModel):
    """`grounded` is True by contract here, unlike /chat. `answer` is null until the
    model gateway is wired (Phase 15) — retrieval is what Phase 8 delivers, and
    returning a fabricated answer to look complete would be worse than null."""

    question: str
    answer: str | None
    citations: list[Citation]
    context_chars: int
    grounded: bool = True
    request_id: str | None = None
    acl_scope: str = ""
    index_version: str = ""
    candidates_considered: int = 0
    retrieval_ms: float = 0.0
    embed_ms: float = 0.0
    search_ms: float = 0.0
    latency_ms: float = 0.0
    mode: str = ""
    dense_candidates: int = 0
    lexical_candidates: int = 0
    reranked: int = 0
    stages: dict[str, float] = {}


# Stub identity. Phase 16 replaces this with a verified JWT from Kong; the shape
# of the claims is already what the token will carry, so the swap is local.
_STUB_USERS = {
    "analyst-emea": User(
        user_id="analyst-emea",
        clearance=Classification.INTERNAL,
        departments=frozenset({"emea-sales", "finance"}),
    ),
    "analyst-apac": User(
        user_id="analyst-apac",
        clearance=Classification.INTERNAL,
        departments=frozenset({"apac-sales"}),
    ),
    "auditor": User(
        user_id="auditor",
        clearance=Classification.RESTRICTED,
        departments=frozenset({"audit", "finance"}),
        valid_from=date(2026, 1, 1),
        valid_until=date(2026, 12, 31),
    ),
    "auditor-expired": User(
        user_id="auditor-expired",
        clearance=Classification.RESTRICTED,
        departments=frozenset({"audit"}),
        valid_from=date(2025, 1, 1),
        valid_until=date(2025, 6, 30),
    ),
    "cfo-office": User(
        user_id="cfo-office",
        clearance=Classification.CONFIDENTIAL,
        departments=frozenset({"corp-dev", "finance", "emea-sales", "apac-sales", "audit"}),
    ),
    "contractor": User(
        user_id="contractor",
        clearance=Classification.PUBLIC,
        departments=frozenset(),
    ),
}


def resolve_user(x_user: str | None) -> User:
    """Fails closed: an unknown or absent identity gets the least-privileged user,
    never a permissive default."""
    if x_user and x_user in _STUB_USERS:
        return _STUB_USERS[x_user]
    return _STUB_USERS["contractor"]


@router.post("/rag", response_model=RagResponse)
async def rag(
    body: RagRequest, request: Request, x_user: str | None = Header(default=None)
) -> RagResponse:
    started = time.perf_counter()
    pipeline: RetrievalPipeline = request.app.state.pipeline
    user = resolve_user(x_user)
    acl = AclPredicate.for_user(user)

    result = pipeline.run(
        body.question,
        acl,
        mode=Mode(body.mode),
        top_k=body.top_k,
        candidates=body.candidates,
    )
    context, _ = build_context(result.chunks)

    return RagResponse(
        question=body.question,
        # Deliberately null: generation arrives with the model gateway in Phase 15.
        answer=None,
        citations=[
            Citation(
                citation=c.citation,
                doc_id=c.doc_id,
                doc_version=c.doc_version,
                title=c.title,
                page=c.page,
                classification=c.classification,
                score=round(c.score, 4),
            )
            for c in result.chunks
        ],
        context_chars=len(context),
        request_id=get_request_id(),
        acl_scope=result.acl_scope,
        index_version=request.app.state.index_version,
        candidates_considered=max(result.dense_candidates, result.lexical_candidates),
        retrieval_ms=result.total_ms,
        embed_ms=result.stages.get("embed_ms", 0.0),
        search_ms=result.stages.get("dense_ms", 0.0),
        latency_ms=round((time.perf_counter() - started) * 1000, 3),
        mode=result.mode,
        dense_candidates=result.dense_candidates,
        lexical_candidates=result.lexical_candidates,
        reranked=result.reranked,
        stages=result.stages,
    )


@router.get("/whoami")
async def whoami(x_user: str | None = Header(default=None)) -> dict[str, object]:
    """Which identity the stub resolved, and the ACL scope it produces. The scope
    hash is what Phase 15 must include in the cache key (ADR-019)."""
    user = resolve_user(x_user)
    acl = AclPredicate.for_user(user)
    return {
        "user_id": user.user_id,
        "clearance": user.clearance.name.lower(),
        "departments": sorted(user.departments),
        "within_validity": user.is_within_validity,
        "acl_scope_hash": acl.scope_hash(),
        "available_stub_users": sorted(_STUB_USERS),
    }


@router.get("/index")
async def index_info(request: Request) -> dict[str, object]:
    return dict(request.app.state.ingest_summary)


def create_app(settings: Settings | None = None) -> FastAPI:
    settings = settings or get_settings()
    configure_logging(
        level=settings.app.log_level, fmt=settings.app.log_format, service=SERVICE_NAME
    )
    health = HealthRegistry(service=SERVICE_NAME)

    @asynccontextmanager
    async def lifespan(app: FastAPI) -> AsyncIterator[None]:
        url = str(settings.qdrant.url)
        client = QdrantClient(url=url)
        embedder = Embedder()

        collections = [c.name for c in client.get_collections().collections]
        docs = sorted(c for c in collections if c.startswith("docs_"))
        if not docs:
            raise NotFoundError(
                f"no docs_* collection in {url}; run `python -m rag_service.ingest`"
            )
        collection = docs[-1]

        payloads = load_payloads(client, collection)
        app.state.client = client
        app.state.retriever = Retriever(client, embedder, collection)
        app.state.pipeline = RetrievalPipeline(
            app.state.retriever, LexicalIndex(payloads), Reranker()
        )
        app.state.index_version = collection.removeprefix("docs_")
        count = client.count(collection_name=collection).count
        app.state.ingest_summary = {
            "collection": collection,
            "index_version": app.state.index_version,
            "chunks": count,
            "embed_model": embedder.model_name,
            "embed_dim": embedder.dim,
            "lexical_chunks": len(payloads),
            "reranker": app.state.pipeline.reranker.model_name,
            "modes": [str(m) for m in Mode],
        }

        async def qdrant_ready() -> None:
            client.get_collections()

        health.register("qdrant", qdrant_ready)
        log.info("startup", **app.state.ingest_summary)
        yield
        client.close()
        log.info("shutdown")

    app = FastAPI(title=SERVICE_NAME, version=__version__, lifespan=lifespan)
    app.add_middleware(RequestContextMiddleware)
    install_error_handlers(app)
    app.include_router(build_health_router(health))
    app.include_router(router)
    app.state.settings = settings
    return app


app = create_app()

if __name__ == "__main__":
    import uvicorn

    uvicorn.run(app, host="0.0.0.0", port=int(os.getenv("PORT", "8070")))  # noqa: S104
