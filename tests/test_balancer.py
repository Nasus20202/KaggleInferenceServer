"""Kaggle-side balancer (kaggle/server.py) against fake llama-server instances."""

import asyncio
import json

from kis.schema import GpuStats, Stats, load

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
    resp = await client.get("/props", headers=AUTH)
    assert (await resp.json())["backend"].startswith("gpu")


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


async def test_keepalive_while_backend_is_slow(aiohttp_client, balancer, server, monkeypatch):
    """Long prompts: bytes start flowing before Cloudflare's first-byte timeout, and clients still parse them."""
    monkeypatch.setattr(server, "KEEPALIVE_SECONDS", 0.05)  # fake backends answer after 0.2 s
    client = await aiohttp_client(balancer.app())

    resp = await client.post("/v1/chat/completions", json={"max_tokens": 4}, headers=AUTH)
    raw = await resp.read()
    assert resp.status == 200 and raw.startswith(b" ")
    assert json.loads(raw)["usage"]["completion_tokens"] == 4

    resp = await client.post("/v1/chat/completions", json={"stream": True}, headers=AUTH)
    lines = [line async for line in resp.content if line.strip()]
    assert lines[0].startswith(b": keepalive") and lines[-1].strip() == b"data: [DONE]"
    assert all(n == 0 for n in balancer.inflight.values())


async def test_stats_need_the_key_and_report_gpus_and_requests(aiohttp_client, balancer, server, monkeypatch):
    monkeypatch.setattr(server, "gpu_stats", lambda: [GpuStats(gpu=0, used_mib=9000, total_mib=15360, util_pct=5)])
    client = await aiohttp_client(balancer.app())
    assert (await client.get("/admin/stats")).status == 401
    await client.post("/v1/chat/completions", json={}, headers=AUTH)
    stats = load(Stats, await (await client.get("/admin/stats", headers=AUTH)).json())
    assert stats.requests == 1 and stats.inflight == 0
    assert stats.gpus[0].used_mib == 9000


async def test_logs_tail_only_log_files(aiohttp_client, balancer, server, monkeypatch, tmp_path):
    monkeypatch.setattr(server, "WORK", tmp_path)
    (tmp_path / "llama-8090.log").write_text("".join(f"line {i}\n" for i in range(10)))
    client = await aiohttp_client(balancer.app())
    resp = await client.get("/admin/logs?file=llama-8090.log&lines=2", headers=AUTH)
    assert await resp.text() == "line 8\nline 9\n"
    for name in ("../secrets.log", "config.toml", "missing.log"):
        resp = await client.get(f"/admin/logs?file={name}", headers=AUTH)
        assert resp.status == 404


async def test_logs_each_request(aiohttp_client, balancer, caplog):
    client = await aiohttp_client(balancer.app())
    with caplog.at_level("INFO", logger="kis"):
        await client.post("/v1/chat/completions", json={}, headers=AUTH)
    assert any("POST /v1/chat/completions -> :" in r.getMessage() and " 200 " in r.getMessage() for r in caplog.records)


async def test_stats_include_token_usage_per_model(aiohttp_client, balancer, server, monkeypatch, caplog):
    monkeypatch.setattr(server, "gpu_stats", list)
    client = await aiohttp_client(balancer.app())
    with caplog.at_level("INFO", logger="kis"):
        for tokens in (3, 5):
            await client.post("/v1/chat/completions", json={"max_tokens": tokens}, headers=AUTH)
    stats = load(Stats, await (await client.get("/admin/stats", headers=AUTH)).json())
    assert stats.usage[server.MODEL.alias].tokens_out == 8
    assert any("out=5" in r.getMessage() for r in caplog.records)


async def test_max_queue_answers_429_when_full(aiohttp_client, balancer, server, monkeypatch):
    monkeypatch.setattr(server.SETTINGS, "max_queue", 1)
    monkeypatch.setattr(server.RUNTIME, "slots", 2)  # 2 slots + 1 waiting = 3 accepted at once
    client = await aiohttp_client(balancer.app())
    resps = await asyncio.gather(*(client.post("/v1/chat/completions", json={}, headers=AUTH) for _ in range(5)))
    assert sorted(r.status for r in resps) == [200, 200, 200, 429, 429]
    busy = next(r for r in resps if r.status == 429)
    assert busy.headers["Retry-After"] == "5"
    monkeypatch.setattr(server.SETTINGS, "max_queue", None)
    resps = await asyncio.gather(*(client.post("/v1/chat/completions", json={}, headers=AUTH) for _ in range(5)))
    assert {r.status for r in resps} == {200}
