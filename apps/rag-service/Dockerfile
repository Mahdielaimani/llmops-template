# syntax=docker/dockerfile:1.7
# Multi-stage: uv resolves the workspace into a venv, runtime carries only the
# venv plus source and runs as non-root.

ARG PYTHON_VERSION=3.12
FROM ghcr.io/astral-sh/uv:0.8.22-python${PYTHON_VERSION}-bookworm-slim AS builder

ENV UV_COMPILE_BYTECODE=1 \
    UV_LINK_MODE=copy \
    UV_PYTHON_DOWNLOADS=never

WORKDIR /build

COPY pyproject.toml uv.lock ./
COPY packages/llmops-core/pyproject.toml packages/llmops-core/
COPY apps/rag-service/pyproject.toml apps/rag-service/
RUN --mount=type=cache,target=/root/.cache/uv \
    uv sync --frozen --no-dev --no-install-workspace --package rag-service

COPY packages/llmops-core packages/llmops-core
COPY apps/rag-service apps/rag-service
RUN --mount=type=cache,target=/root/.cache/uv \
    uv sync --frozen --no-dev --package rag-service

# ---------------------------------------------------------------------------
FROM python:${PYTHON_VERSION}-slim-bookworm AS runtime

# `app` is a system user with no writable home, so both cache roots must be
# explicit or huggingface_hub fails with EACCES when pre-downloading the model.
ENV PYTHONUNBUFFERED=1 \
    PYTHONDONTWRITEBYTECODE=1 \
    PATH="/build/.venv/bin:$PATH" \
    FASTEMBED_CACHE_PATH=/build/.cache/fastembed \
    HF_HOME=/build/.cache/hf \
    HF_HUB_DISABLE_SYMLINKS_WARNING=1 \
    HF_HUB_DISABLE_TELEMETRY=1

RUN groupadd --system app && useradd --system --gid app --uid 10001 app \
    && apt-get update && apt-get install -y --no-install-recommends curl \
    && rm -rf /var/lib/apt/lists/*

WORKDIR /build
COPY --from=builder --chown=app:app /build /build
RUN mkdir -p /build/.cache/fastembed /build/.cache/hf && chown -R app:app /build/.cache
USER app

# Bake the ONNX weights in: otherwise the first request pays an ~11 s download
# and the readiness probe flaps while the model is fetched.
RUN python -c "from fastembed import TextEmbedding; TextEmbedding('BAAI/bge-small-en-v1.5')"

EXPOSE 8070
HEALTHCHECK --interval=10s --timeout=3s --start-period=15s --retries=3 \
    CMD curl -fsS http://localhost:8070/health/live || exit 1

CMD ["uvicorn", "rag_service.main:app", "--host", "0.0.0.0", "--port", "8070"]
