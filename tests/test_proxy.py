"""Local proxy (kis/proxy.py) chained to the Kaggle-side balancer and fake llama-servers."""

import dataclasses

import pytest
from conftest import SECRETS

from kis import events, proxy


@pytest.fixture
async def balancer_url(aiohttp_server, balancer):
    server = await aiohttp_server(balancer.app())
    return str(server.make_url("")).rstrip("/")


async def make_client(aiohttp_client, monkeypatch, config, url):
    return await make_client_with(aiohttp_client, monkeypatch, config, url, SECRETS)


async def make_client_with(aiohttp_client, monkeypatch, config, url, secrets):
    monkeypatch.setattr(events, "endpoint", lambda config, topic: url)
    return await aiohttp_client(proxy.make_app(config, secrets))


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


async def test_logs_each_request(aiohttp_client, monkeypatch, config, balancer_url, caplog):
    client = await make_client(aiohttp_client, monkeypatch, config, balancer_url)
    with caplog.at_level("INFO", logger="kis.proxy"):
        await client.get("/v1/models")
    assert any(r.getMessage().startswith("GET /v1/models 200 ") for r in caplog.records)


async def test_retries_transient_upstream_errors(aiohttp_client, aiohttp_server, monkeypatch, config):
    from aiohttp import web

    calls = []

    async def models(req: web.Request) -> web.Response:
        calls.append(1)
        if len(calls) < 3:
            return web.Response(status=502, text="tunnel reconnecting")
        return web.json_response({"data": [{"id": "m"}]})

    app = web.Application()
    app.router.add_get("/v1/models", models)
    upstream = await aiohttp_server(app)
    client = await make_client(aiohttp_client, monkeypatch, config, str(upstream.make_url("")).rstrip("/"))
    resp = await client.get("/v1/models")
    assert resp.status == 200 and len(calls) == 3


async def test_does_not_retry_client_errors(aiohttp_client, monkeypatch, config, balancer_url):
    wrong = dataclasses.replace(SECRETS, api_key="wrong")
    client = await make_client_with(aiohttp_client, monkeypatch, config, balancer_url, wrong)
    assert (await client.get("/v1/models")).status == 401


async def test_optional_local_api_key(aiohttp_client, monkeypatch, config, balancer_url):
    monkeypatch.setenv("KIS_PROXY_API_KEY", "local-secret")
    client = await make_client(aiohttp_client, monkeypatch, config, balancer_url)
    assert (await client.get("/v1/models")).status == 401
    assert (await client.get("/v1/models", headers={"Authorization": "Bearer nope"})).status == 401
    assert (await client.get("/v1/models", headers={"Authorization": "Bearer local-secret"})).status == 200
    assert (await client.get("/v1/models", headers={"x-api-key": "local-secret"})).status == 200
