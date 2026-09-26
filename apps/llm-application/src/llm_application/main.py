from __future__ import annotations

from collections.abc import AsyncIterator
from contextlib import asynccontextmanager

from fastapi import FastAPI

from llmops_core import __version__
from llmops_core.config import Settings, get_settings
from llmops_core.context import RequestContextMiddleware
from llmops_core.errors import install_error_handlers
from llmops_core.health import HealthRegistry, build_health_router
from llmops_core.logging import configure_logging, get_logger

log = get_logger("llm_application")


def create_app(settings: Settings | None = None) -> FastAPI:
    settings = settings or get_settings()
    configure_logging(
        level=settings.app.log_level, fmt=settings.app.log_format, service=settings.app.name
    )

    health = HealthRegistry(service=settings.app.name)

    @asynccontextmanager
    async def lifespan(_: FastAPI) -> AsyncIterator[None]:
        log.info(
            "startup",
            env=settings.app.env,
            llm_provider=settings.llm.provider,
            llm_model=settings.llm.model,
            port=settings.server.port,
        )
        yield
        log.info("shutdown")

    app = FastAPI(
        title=settings.app.name,
        version=__version__,
        lifespan=lifespan,
        docs_url="/docs" if settings.app.env != "prod" else None,
    )
    app.add_middleware(RequestContextMiddleware)
    install_error_handlers(app)
    app.include_router(build_health_router(health))
    app.state.settings = settings
    return app


app = create_app()
