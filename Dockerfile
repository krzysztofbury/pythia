# syntax=docker/dockerfile:1.7

FROM ghcr.io/astral-sh/uv:0.12.10 AS uv

# Digest-pinned so the smoke-tested base cannot drift under a tag between
# CI jobs; Dependabot's docker ecosystem updates the pin.
FROM python:3.13-slim-bookworm@sha256:9d7f287598e1a5a978c015ee176d8216435aaf335ed69ac3c38dd1bbb10e8d64 AS builder

ENV UV_COMPILE_BYTECODE=1 \
    UV_LINK_MODE=copy \
    UV_PYTHON_DOWNLOADS=never \
    UV_PROJECT_ENVIRONMENT=/app/.venv

WORKDIR /app

COPY --from=uv /uv /usr/local/bin/uv
COPY pyproject.toml uv.lock ./
RUN uv sync --frozen --no-dev --no-install-project

FROM python:3.13-slim-bookworm@sha256:9d7f287598e1a5a978c015ee176d8216435aaf335ed69ac3c38dd1bbb10e8d64 AS runtime

ARG BUILD_DATE=unknown
ARG VERSION=dev
ARG VCS_REF=unknown

LABEL org.opencontainers.image.created="${BUILD_DATE}" \
      org.opencontainers.image.description="Oracle Pythia crypto trading bot" \
      org.opencontainers.image.licenses="MIT" \
      org.opencontainers.image.revision="${VCS_REF}" \
      org.opencontainers.image.source="https://github.com/krzysztofbury/pythia" \
      org.opencontainers.image.title="Oracle Pythia" \
      org.opencontainers.image.version="${VERSION}"

ENV BUILD_VERSION="${VERSION}" \
    HEALTHCHECK_FILE=/app/logs/healthcheck \
    HEALTHCHECK_MAX_AGE_SECONDS=2700 \
    HOME=/home/oracle \
    MPLCONFIGDIR=/tmp/matplotlib \
    NUMBA_CACHE_DIR=/tmp/numba \
    PATH=/app/.venv/bin:$PATH \
    PYTHONDONTWRITEBYTECODE=1 \
    PYTHONPATH=/app/src \
    PYTHONUNBUFFERED=1 \
    XDG_CACHE_HOME=/tmp/.cache

RUN apt-get update \
    && apt-get install --no-install-recommends --yes ca-certificates libgomp1 \
    && rm -rf /var/lib/apt/lists/* \
    && groupadd --gid 10001 oracle \
    && useradd --create-home --gid 10001 --no-log-init --uid 10001 oracle \
    && mkdir -p /app/logs \
    && chown oracle:oracle /app/logs

WORKDIR /app

COPY --from=builder --chown=oracle:oracle /app/.venv /app/.venv
COPY --chown=oracle:oracle src/ /app/src/

USER 10001:10001

HEALTHCHECK --interval=5m --timeout=10s --start-period=2m --retries=3 \
    CMD ["/bin/sh", "-c", "exec /app/.venv/bin/python /app/src/health.py \"${HEALTHCHECK_FILE:-/app/logs/healthcheck}\" --max-age-seconds \"${HEALTHCHECK_MAX_AGE_SECONDS:-2700}\""]

CMD ["/app/.venv/bin/python", "/app/src/main.py"]
