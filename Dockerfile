# kis client: proxy, up/down/bench and the Kaggle CLI. The GPU side runs on Kaggle.
FROM python:3.15.2-slim@sha256:63bb298d2b7b88f176f51b5189f54ce6535c997682ea70885d2132f5d473a8a5

COPY --from=ghcr.io/astral-sh/uv:0.13.10@sha256:a3ea1357342c1a4bf9a0c53a3deb1b55379aefa037338d8113d5771efa01774f /uv /bin/uv
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
