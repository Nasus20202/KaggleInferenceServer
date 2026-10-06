"""Kaggle-side balancer (kaggle/server.py) against fake llama-server instances."""

import asyncio

AUTH = {"Authorization": "Bearer secret"}


async def test_health_needs_no_key(aiohttp_client, balancer):
    client = await aiohttp_client(balancer.app())
    resp = await client.get("/health")
    assert resp.status == 200


async def test_rejects_missing_or_wrong_key(aiohttp_client, balancer):
    client = await aiohttp_client(balancer.app())
    for headers in ({}, {"Authorization": "Bearer wrong"}):
        resp = await client.post("/v1/chat/completions", json={}, headers=headers)
        assert resp.status == 401


async def test_forwards_without_api_key(aiohttp_client, balancer, llama_servers):
    client = await aiohttp_client(balancer.app())
    resp = await client.post("/v1/chat/completions", json={"max_tokens": 3}, headers=AUTH)
    assert resp.status == 200
    assert (await resp.json())["usage"]["completion_tokens"] == 3
    received = [r for s in llama_servers for r in s.app["received"]]
    assert len(received) == 1
    assert "Authorization" not in received[0]["headers"]


async def test_spreads_concurrent_requests_over_instances(aiohttp_client, balancer, llama_servers):
    client = await aiohttp_client(balancer.app())

    async def one():
        resp = await client.post("/v1/chat/completions", json={}, headers=AUTH)
        return (await resp.json())["backend"]

    backends = await asyncio.gather(*(one() for _ in range(6)))
    assert sorted(backends) == ["gpu0"] * 3 + ["gpu1"] * 3
    assert balancer.inflight == {s.port: 0 for s in llama_servers}


async def test_passes_through_backend_routes(aiohttp_client, balancer):
    client = await aiohttp_client(balancer.app())
    resp = await client.get("/v1/models", headers=AUTH)
    assert (await resp.json())["data"][0]["id"].startswith("gpu")


async def test_streams_sse(aiohttp_client, balancer):
    client = await aiohttp_client(balancer.app())
    resp = await client.post("/v1/chat/completions", json={"stream": True}, headers=AUTH)
    assert resp.headers["Content-Type"].startswith("text/event-stream")
    events = [line async for line in resp.content if line.startswith(b"data:")]
    assert len(events) == 4
    assert events[-1].strip() == b"data: [DONE]"


async def test_client_disconnect_mid_stream_frees_slot(aiohttp_client, balancer):
    client = await aiohttp_client(balancer.app())
    resp = await client.post("/v1/chat/completions", json={"stream": True}, headers=AUTH)
    await resp.content.readline()
    resp.close()
    await asyncio.sleep(0.3)
    assert not any(balancer.inflight.values())


async def test_idle_seconds_is_zero_while_busy(aiohttp_client, balancer):
    client = await aiohttp_client(balancer.app())
    request = asyncio.create_task(client.post("/v1/chat/completions", json={}, headers=AUTH))
    await asyncio.sleep(0.1)
    assert balancer.idle_seconds == 0
    await request
    assert 0 <= balancer.idle_seconds < 1


async def test_shutdown_requires_key_and_sets_stop(aiohttp_client, balancer):
    client = await aiohttp_client(balancer.app())
    assert (await client.post("/admin/shutdown")).status == 401
    assert not balancer.stop.is_set()
    assert (await client.post("/admin/shutdown", headers=AUTH)).status == 200
    assert balancer.stop.is_set()


def test_plan_replicates_small_models_and_splits_large(server):
    assert server.plan(["0", "1"], 5.9, None, 11.0) == ("replicas", [["0"], ["1"]])
    assert server.plan(["0", "1"], 17.1, None, 11.0) == ("split", [["0", "1"]])
    assert server.plan(["0", "1"], 5.9, "split", 11.0) == ("split", [["0", "1"]])
    assert server.plan(["0"], 5.9, None, 11.0) == ("replicas", [["0"]])
