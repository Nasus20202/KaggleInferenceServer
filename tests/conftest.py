import asyncio
import importlib.util
import json
import sys
import time
import tomllib
from pathlib import Path

import pytest
from aiohttp import web

from kis.settings import Config, Secrets, parse_config

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT / "kaggle"))  # the Kaggle scripts import their modules from kaggle/ (inlined when pushed)


@pytest.fixture(scope="session")
def server():
    """kaggle/server.py as a module (it is a Kaggle script, not part of the kis package)."""
    spec = importlib.util.spec_from_file_location("kaggle_server", ROOT / "kaggle" / "server.py")
    assert spec and spec.loader
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


@pytest.fixture
def config() -> Config:
    data = tomllib.loads((ROOT / "examples" / "config.32k.toml").read_text())
    data["kaggle"]["username"] = "tester"
    return parse_config(data, environ={})


SECRETS = Secrets(api_key="secret", ntfy_topic="test")


def fake_llama_server(name: str, delay: float = 0.0) -> web.Application:
    """Stand-in for llama-server: JSON or SSE chat completions; records what it received."""
    app = web.Application()
    app["received"] = []

    async def chat(req: web.Request) -> web.StreamResponse:
        body = await req.json()
        app["received"].append({"headers": dict(req.headers), "body": body})
        await asyncio.sleep(delay)
        if body.get("stream"):
            resp = web.StreamResponse(headers={"Content-Type": "text/event-stream"})
            await resp.prepare(req)
            for i in range(3):
                await resp.write(f"data: {json.dumps({'backend': name, 'i': i})}\n\n".encode())
            await resp.write(b"data: [DONE]\n\n")
            return resp
        tokens = body.get("max_tokens", 8)
        return web.json_response(
            {
                "backend": name,
                "choices": [{"message": {"role": "assistant", "content": "hi"}}],
                "usage": {"completion_tokens": tokens},
                "timings": {"predicted_per_second": 50.0, "draft_n": 10, "draft_n_accepted": 7},
            }
        )

    async def models(_req: web.Request) -> web.Response:
        return web.json_response({"data": [{"id": name}]})

    async def props(_req: web.Request) -> web.Response:
        return web.json_response({"backend": name})

    app.router.add_post("/v1/chat/completions", chat)
    app.router.add_post("/v1/embeddings", chat)
    app.router.add_get("/v1/models", models)
    app.router.add_get("/props", props)
    return app


class FakeBackends:
    """Stand-in for server.Backends: the ports of fake llama-servers per preset. Downloads
    take `fetch_s`; loading a preset in `fail` raises."""

    def __init__(self, ports: dict[str, list[int]], loaded: str, fetch_s: float = 0.0, fail: tuple[str, ...] = ()):
        self.by_preset, self.loaded = ports, loaded
        self.fetch_s, self.fail = fetch_s, fail
        self.fetched: list[str] = [loaded]
        self.loads: list[str] = []

    @property
    def ports(self) -> list[int]:
        return self.by_preset[self.loaded]

    def fetch(self, name: str) -> None:
        time.sleep(self.fetch_s)
        self.fetched.append(name)

    def downloaded(self, name: str) -> bool:
        return name in self.fetched

    def load(self, name: str) -> list[int]:
        self.loads.append(name)
        if name in self.fail:
            raise RuntimeError(f"cannot load {name}")
        self.loaded = name
        return self.ports


@pytest.fixture
async def llama_servers(aiohttp_server):
    """Two fake llama-server instances, slow enough that concurrent requests overlap."""
    return [await aiohttp_server(fake_llama_server(f"gpu{i}", delay=0.2)) for i in range(2)]


def make_balancer(server, backends, presets=None, loaded=None, residents=None):
    """A Balancer wired to the settings, runtime and notify of `server` (read when called, so
    tests can patch them first), with the api key "secret"."""
    from balancer import Balancer  # on sys.path since the line above

    return Balancer(
        backends,
        "secret",
        server.SETTINGS,
        server.RUNTIME,
        server.notify,
        presets or server.PRESETS,
        loaded or server.START,
        residents,
    )


@pytest.fixture
async def balancer(server, llama_servers):
    return make_balancer(server, FakeBackends({server.START: [s.port for s in llama_servers]}, server.START))


@pytest.fixture(autouse=True)
def no_retry_delay(monkeypatch):
    from kis import retry

    monkeypatch.setattr(retry, "DELAYS", (0.0,))
