# syntax=docker/dockerfile:1

FROM python:3.12-slim AS runtime

# onnxruntime (fastembed) needs libgomp; curl is used by the compose healthcheck.
RUN apt-get update && apt-get install -y --no-install-recommends \
        libgomp1 curl \
    && rm -rf /var/lib/apt/lists/*

# uv: fast, reproducible Python dependency management.
COPY --from=ghcr.io/astral-sh/uv:0.11 /uv /uvx /bin/

WORKDIR /app

ENV UV_COMPILE_BYTECODE=1 \
    UV_LINK_MODE=copy \
    PYTHONUNBUFFERED=1 \
    PYTHONPATH=/app/src \
    # Cache fastembed/HF weights inside the image (not a volume) so retrieval
    # works with no network at runtime.
    FASTEMBED_CACHE_PATH=/opt/models \
    HF_HOME=/opt/models/hf \
    EMBED_MODEL=BAAI/bge-small-en-v1.5 \
    RERANK_MODEL=Xenova/ms-marco-MiniLM-L-6-v2 \
    PORT=8080

# 1) Dependencies first (best layer caching). This is a virtual uv project
#    (no build-system), so syncing deps does not require the source tree.
COPY pyproject.toml uv.lock ./
RUN uv sync --no-dev --frozen

# 2) Bake embedding + reranker models into the image.
COPY scripts/prefetch_models.py ./scripts/prefetch_models.py
RUN uv run python scripts/prefetch_models.py

# 3) Application source (changes most often -> last).
COPY src ./src

EXPOSE 8080

# Container-level healthcheck mirrors the eval's readiness probe (honors $PORT).
HEALTHCHECK --interval=5s --timeout=3s --start-period=20s --retries=12 \
    CMD curl -sf http://localhost:${PORT:-8080}/health || exit 1

# Shell form so $PORT is expanded at runtime (defaults to 8080).
CMD uv run --no-dev uvicorn memory_service.main:app --host 0.0.0.0 --port ${PORT:-8080} --workers 1
