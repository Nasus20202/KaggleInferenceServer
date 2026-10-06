"""Local proxy (kis/proxy.py) chained to the Kaggle-side balancer and fake llama-servers."""

import pytest

from kis import events, proxy

SECRETS = {"api_key": "secret", "ntfy_topic": "test"}


@pytest.fixture
async def balancer_url(aiohttp_server, balancer):
    server = await aiohttp_server(balancer.app())
    return str(server.make_url("")).rstrip("/")


async def make_client(aiohttp_client, monkeypatch, config, url):
    monkeypatch.setattr(events, "endpoint", lambda config, topic: url)
    return await aiohttp_client(proxy.make_app(config, SECRETS))


async def test_adds_api_key(aiohttp_client, monkeypatch, config, balancer_url):
    client = await make_client(aiohttp_client, monkeypatch, config, balancer_url)
    resp = await client.post(
        "/v1/chat/completions", json={"max_tokens": 5}, headers={"Authorization": "Bearer anything"}
    )
    assert resp.status == 200
    assert (await resp.json())["usage"]["completion_tokens"] == 5


async def test_streams_end_to_end(aiohttp_client, monkeypatch, config, balancer_url):
    client = await make_client(aiohttp_client, monkeypatch, config, balancer_url)
    resp = await client.post("/v1/chat/completions", json={"stream": True})
    events_ = [line async for line in resp.content if line.startswith(b"data:")]
    assert events_[-1].strip() == b"data: [DONE]"


async def test_no_server_running(aiohttp_client, monkeypatch, config):
    client = await make_client(aiohttp_client, monkeypatch, config, None)
    resp = await client.get("/v1/models")
    assert resp.status == 503


async def test_server_unreachable(aiohttp_client, monkeypatch, config, unused_tcp_port):
    client = await make_client(aiohttp_client, monkeypatch, config, f"http://127.0.0.1:{unused_tcp_port}")
    resp = await client.get("/v1/models")
    assert resp.status == 502


async def test_decompresses_gzip_from_the_tunnel(aiohttp_client, aiohttp_server, monkeypatch, config):
    """Cloudflare gzips responses; the proxy must not pass compressed bytes without Content-Encoding."""
    from aiohttp import web

    async def models(req: web.Request) -> web.Response:
        resp = web.json_response({"data": [{"id": "m"}]})
        resp.enable_compression(web.ContentCoding.gzip)
        return resp

    app = web.Application()
    app.router.add_get("/v1/models", models)
    upstream = await aiohttp_server(app)
    client = await make_client(aiohttp_client, monkeypatch, config, str(upstream.make_url("")).rstrip("/"))
    resp = await client.get("/v1/models", headers={"Accept-Encoding": "identity"})
    assert "Content-Encoding" not in resp.headers
    assert (await resp.json())["data"][0]["id"] == "m"
