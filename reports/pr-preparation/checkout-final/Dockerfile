# syntax=docker/dockerfile:1

FROM python:3.12-slim-bookworm

# Pin uv so dependency installation is reproducible across image rebuilds.
COPY --from=ghcr.io/astral-sh/uv:0.11.9 /uv /uvx /bin/

ENV PYTHONDONTWRITEBYTECODE=1 \
    PYTHONUNBUFFERED=1 \
    UV_COMPILE_BYTECODE=1 \
    UV_LINK_MODE=copy \
    UV_NO_DEV=1

WORKDIR /app

# Keep dependency installation cached until the project metadata changes.
COPY pyproject.toml uv.lock README.md .python-version ./
RUN --mount=type=cache,target=/root/.cache/uv \
    uv sync --locked --no-install-project

COPY . .
RUN --mount=type=cache,target=/root/.cache/uv \
    uv sync --locked && \
    groupadd --system app && \
    useradd --system --gid app --home-dir /app app && \
    chown -R app:app /app

USER app

EXPOSE 8080

CMD ["uv", "run", "--no-sync", "uvicorn", "market.dashboard:app", "--host", "0.0.0.0", "--port", "8080"]
