import asyncio
import importlib.util
import json
import tomllib
from pathlib import Path

import pytest
from aiohttp import web

ROOT = Path(__file__).resolve().parent.parent


@pytest.fixture(scope="session")
def server():
    """kaggle/server.py as a module (it is a Kaggle script, not part of the kis package)."""
    spec = importlib.util.spec_from_file_location("kaggle_server", ROOT / "kaggle" / "server.py")
    assert spec and spec.loader
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


@pytest.fixture
def config():
    data = tomllib.loads((ROOT / "examples" / "config.32k.toml").read_text())
    data["kaggle"]["username"] = "tester"
    return data


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

    app.router.add_post("/v1/chat/completions", chat)
    app.router.add_get("/v1/models", models)
    return app


@pytest.fixture
async def llama_servers(aiohttp_server):
    """Two fake llama-server instances, slow enough that concurrent requests overlap."""
    return [await aiohttp_server(fake_llama_server(f"gpu{i}", delay=0.2)) for i in range(2)]


@pytest.fixture
async def balancer(server, llama_servers):
    return server.Balancer([s.port for s in llama_servers], api_key="secret")


@pytest.fixture(autouse=True)
def no_retry_delay(monkeypatch):
    from kis import retry

    monkeypatch.setattr(retry, "DELAYS", (0.0,))
