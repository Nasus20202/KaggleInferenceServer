# kis client: proxy, up/down/bench and the Kaggle CLI. The GPU side runs on Kaggle.
FROM python:3.14.8-slim

COPY --from=ghcr.io/astral-sh/uv:0.13.0 /uv /bin/uv
ENV UV_COMPILE_BYTECODE=1 UV_LINK_MODE=copy UV_PYTHON_DOWNLOADS=never PYTHONUNBUFFERED=1

RUN useradd --create-home --uid 1000 kis
WORKDIR /app

COPY pyproject.toml uv.lock ./
RUN uv sync --frozen --no-dev --no-install-project
COPY kis ./kis
COPY kaggle ./kaggle
COPY README.md ./
COPY examples ./examples
RUN uv sync --frozen --no-dev && mkdir -p /app/.kis && chown kis:kis /app/.kis

# config.toml -> /app/config.toml, state -> /app/.kis, Kaggle token -> /home/kis/.kaggle
ENV PATH="/app/.venv/bin:$PATH"
USER kis
EXPOSE 8080
ENTRYPOINT ["kis"]
CMD ["proxy", "--host", "0.0.0.0", "--port", "8080"]
