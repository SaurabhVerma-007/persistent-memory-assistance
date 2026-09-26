FROM python:3.12-slim

ENV PYTHONDONTWRITEBYTECODE=1 \
    PYTHONUNBUFFERED=1 \
    UV_COMPILE_BYTECODE=1 \
    UV_LINK_MODE=copy

WORKDIR /app

COPY --from=ghcr.io/astral-sh/uv:latest /uv /uvx /bin/
COPY pyproject.toml uv.lock ./
COPY README.md ./README.md
COPY mem ./mem
COPY local-qdrant/pyproject.toml ./local-qdrant/pyproject.toml
COPY local-qdrant/README.md ./local-qdrant/README.md
COPY local-qdrant/src ./local-qdrant/src
RUN uv sync --frozen --no-dev

COPY static ./static
COPY web_app.py main.py basic_mem0_chatbot.py ./

EXPOSE 8000
CMD ["sh", "-c", "uv run --no-dev uvicorn web_app:app --host 0.0.0.0 --port ${PORT:-8000}"]
