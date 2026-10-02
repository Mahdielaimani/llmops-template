from __future__ import annotations

import time
from collections.abc import AsyncIterator
from contextlib import asynccontextmanager

from fastapi import APIRouter, FastAPI, Request
from pydantic import BaseModel, Field

from classical_ml.model import Invoice, LoadedModel, Prediction, load_model
from llmops_core import __version__
from llmops_core.config import Settings, get_settings
from llmops_core.context import RequestContextMiddleware, get_request_id
from llmops_core.errors import install_error_handlers
from llmops_core.health import HealthRegistry, build_health_router
from llmops_core.logging import configure_logging, get_logger

log = get_logger("classical_ml")
router = APIRouter(tags=["predict"])

SERVICE_NAME = "classical-ml"


class BatchRequest(BaseModel):
    """Batching here is a caller convenience, not a scheduler. Contrast vLLM's
    continuous batching, where the engine interleaves requests token by token."""

    invoices: list[Invoice] = Field(min_length=1, max_length=1000)


class BatchResponse(BaseModel):
    predictions: list[Prediction]
    count: int
    latency_ms: float


@router.post("/predict", response_model=Prediction)
async def predict(invoice: Invoice, request: Request) -> Prediction:
    model: LoadedModel = request.app.state.model
    started = time.perf_counter()
    labels, probas = model.predict([invoice])
    latency = round((time.perf_counter() - started) * 1000, 3)
    log.info(
        "predict",
        model_name=model.name,
        model_version=model.version,
        probability=probas[0],
        latency_ms=latency,
    )
    return Prediction(
        paid_late=labels[0],
        probability=probas[0],
        threshold=model.threshold,
        model_name=model.name,
        model_version=model.version,
        request_id=get_request_id(),
        latency_ms=latency,
    )


@router.post("/predict/batch", response_model=BatchResponse)
async def predict_batch(body: BatchRequest, request: Request) -> BatchResponse:
    model: LoadedModel = request.app.state.model
    started = time.perf_counter()
    labels, probas = model.predict(body.invoices)
    latency = round((time.perf_counter() - started) * 1000, 3)
    rid = get_request_id()
    log.info("predict_batch", model_version=model.version, count=len(labels), latency_ms=latency)
    return BatchResponse(
        predictions=[
            Prediction(
                paid_late=label,
                probability=proba,
                threshold=model.threshold,
                model_name=model.name,
                model_version=model.version,
                request_id=rid,
                latency_ms=latency,
            )
            for label, proba in zip(labels, probas, strict=True)
        ],
        count=len(labels),
        latency_ms=latency,
    )


@router.get("/model")
async def model_card(request: Request) -> dict[str, object]:
    """The artifact's own metadata, served. An LLM endpoint cannot answer
    "what exactly is serving me" this precisely without a registry."""
    model: LoadedModel = request.app.state.model
    return {"threshold": model.threshold, **model.metadata}


def create_app(settings: Settings | None = None) -> FastAPI:
    settings = settings or get_settings()
    configure_logging(
        level=settings.app.log_level, fmt=settings.app.log_format, service=SERVICE_NAME
    )
    health = HealthRegistry(service=SERVICE_NAME)

    @asynccontextmanager
    async def lifespan(app: FastAPI) -> AsyncIterator[None]:
        started = time.perf_counter()
        app.state.model = load_model()
        load_ms = round((time.perf_counter() - started) * 1000, 2)
        log.info(
            "startup",
            model_version=app.state.model.version,
            artifact_bytes=app.state.model.metadata["artifact_bytes"],
            model_load_ms=load_ms,
        )
        yield
        log.info("shutdown")

    app = FastAPI(title=SERVICE_NAME, version=__version__, lifespan=lifespan)
    app.add_middleware(RequestContextMiddleware)
    install_error_handlers(app)
    app.include_router(build_health_router(health))
    app.include_router(router)
    app.state.settings = settings
    return app


app = create_app()
