# syntax=docker/dockerfile:1.7
# Multi-stage build: uv resolves the workspace into a self-contained venv,
# the runtime stage carries only the venv + source, runs as non-root.

ARG PYTHON_VERSION=3.12
FROM ghcr.io/astral-sh/uv:0.8.22-python${PYTHON_VERSION}-bookworm-slim AS builder

ENV UV_COMPILE_BYTECODE=1 \
    UV_LINK_MODE=copy \
    UV_PYTHON_DOWNLOADS=never

WORKDIR /build

# Layer 1: dependency metadata only → cache survives source edits.
COPY pyproject.toml uv.lock ./
COPY packages/llmops-core/pyproject.toml packages/llmops-core/
COPY apps/llm-application/pyproject.toml apps/llm-application/
RUN --mount=type=cache,target=/root/.cache/uv \
    uv sync --frozen --no-dev --no-install-workspace --package llm-application

# Layer 2: source.
COPY packages/llmops-core packages/llmops-core
COPY apps/llm-application apps/llm-application
RUN --mount=type=cache,target=/root/.cache/uv \
    uv sync --frozen --no-dev --package llm-application

# ---------------------------------------------------------------------------
FROM python:${PYTHON_VERSION}-slim-bookworm AS runtime

ENV PYTHONUNBUFFERED=1 \
    PYTHONDONTWRITEBYTECODE=1 \
    PATH="/build/.venv/bin:$PATH"

RUN groupadd --system app && useradd --system --gid app --uid 10001 app \
    && apt-get update && apt-get install -y --no-install-recommends curl \
    && rm -rf /var/lib/apt/lists/*

WORKDIR /build
COPY --from=builder --chown=app:app /build /build
USER app

EXPOSE 8080
HEALTHCHECK --interval=10s --timeout=3s --start-period=5s --retries=3 \
    CMD curl -fsS http://localhost:8080/health/live || exit 1

CMD ["uvicorn", "llm_application.main:app", "--host", "0.0.0.0", "--port", "8080"]
