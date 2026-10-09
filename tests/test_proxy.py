"""Local proxy (kis/proxy.py) chained to the Kaggle-side balancer and fake llama-servers."""

import asyncio
import contextlib
import dataclasses
import json

import aiohttp
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


async def broken_upstream(aiohttp_server, first_bytes: list[bytes], then_ok: int):
    """An upstream whose first `len(first_bytes)` calls send 200 and these bytes, then reset the
    connection; the calls after that answer with a JSON body. Returns (server, calls)."""
    from aiohttp import web

    calls: list[int] = []

    async def chat(req: web.Request) -> web.StreamResponse:
        calls.append(1)
        if len(calls) <= len(first_bytes):
            resp = web.StreamResponse()
            await resp.prepare(req)
            await resp.write(first_bytes[len(calls) - 1])
            await asyncio.sleep(0.05)
            assert req.transport
            req.transport.abort()
            return resp
        return web.json_response({"ok": True})

    app = web.Application()
    app.router.add_post("/v1/chat/completions", chat)
    return await aiohttp_server(app), calls


async def test_retries_when_the_connection_breaks_before_the_answer(
    aiohttp_client, aiohttp_server, monkeypatch, config
):
    """The balancer starts a slow request's 200 with padding; a reset after only that is retried
    and the answer continues the same response."""
    upstream, calls = await broken_upstream(aiohttp_server, [b" ", b"  "], then_ok=1)
    client = await make_client(aiohttp_client, monkeypatch, config, str(upstream.make_url("")).rstrip("/"))
    resp = await client.post("/v1/chat/completions", json={})
    body = await resp.text()  # the headers come first; the retries happen while the body is read
    assert resp.status == 200 and len(calls) == 3
    assert json.loads(body) == {"ok": True}


async def test_retries_an_event_stream_that_only_sent_keepalives(aiohttp_client, aiohttp_server, monkeypatch, config):
    upstream, calls = await broken_upstream(aiohttp_server, [b": keepalive\n\n"], then_ok=1)
    client = await make_client(aiohttp_client, monkeypatch, config, str(upstream.make_url("")).rstrip("/"))
    resp = await client.post("/v1/chat/completions", json={})
    body = await resp.text()
    assert len(calls) == 2 and body.startswith(": keepalive") and json.loads(body.split("\n\n", 1)[1]) == {"ok": True}


async def test_does_not_retry_after_part_of_the_answer(aiohttp_client, aiohttp_server, monkeypatch, config):
    upstream, calls = await broken_upstream(aiohttp_server, [b'{"choices": [{"mess'], then_ok=1)
    client = await make_client(aiohttp_client, monkeypatch, config, str(upstream.make_url("")).rstrip("/"))
    resp = await client.post("/v1/chat/completions", json={})
    with contextlib.suppress(aiohttp.ClientPayloadError):  # the client sees the cut-off body too
        await resp.text()
    assert len(calls) == 1


async def test_gives_up_with_an_error_body_after_only_padding(aiohttp_client, aiohttp_server, monkeypatch, config):
    upstream, calls = await broken_upstream(aiohttp_server, [b" "] * proxy.ATTEMPTS, then_ok=0)
    client = await make_client(aiohttp_client, monkeypatch, config, str(upstream.make_url("")).rstrip("/"))
    resp = await client.post("/v1/chat/completions", json={})
    body = await resp.text()
    assert resp.status == 200 and len(calls) == proxy.ATTEMPTS
    assert "unreachable" in json.loads(body)["error"]["message"]
