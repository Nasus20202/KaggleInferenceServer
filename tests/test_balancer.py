"""Kaggle-side balancer (kaggle/balancer.py) against fake llama-server instances."""

import asyncio
import json

import balancer as balancer_mod
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


async def test_keepalive_while_backend_is_slow(aiohttp_client, balancer, server, monkeypatch):
    """Long prompts: bytes start flowing before Cloudflare's first-byte timeout, and clients still parse them."""
    monkeypatch.setattr(balancer_mod, "KEEPALIVE_SECONDS", 0.05)  # fake backends answer after 0.2 s
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
    monkeypatch.setattr(
        balancer_mod, "gpu_stats", lambda: [GpuStats(gpu=0, used_mib=9000, total_mib=15360, util_pct=5)]
    )
    client = await aiohttp_client(balancer.app())
    assert (await client.get("/admin/stats")).status == 401
    await client.post("/v1/chat/completions", json={}, headers=AUTH)
    stats = load(Stats, await (await client.get("/admin/stats", headers=AUTH)).json())
    assert stats.requests == 1 and stats.inflight == 0
    assert stats.gpus[0].used_mib == 9000


async def test_logs_tail_only_log_files(aiohttp_client, balancer, server, monkeypatch, tmp_path):
    monkeypatch.setattr(balancer_mod, "WORK", tmp_path)
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
    monkeypatch.setattr(balancer_mod, "gpu_stats", list)
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


def chat(*messages, tools=None):
    return json.dumps({"model": "m", "messages": list(messages), "tools": tools or []}).encode()


SYSTEM = {"role": "system", "content": "You fix clusters."}


def test_affinity_key_is_the_same_for_every_turn_of_a_conversation():
    first = {"role": "user", "content": "pod is pending"}
    turn1 = chat(SYSTEM, first)
    turn2 = chat(
        SYSTEM, first, {"role": "assistant", "content": "kubectl get pods"}, {"role": "tool", "content": "..."}
    )
    assert balancer_mod.affinity_key(turn1) == balancer_mod.affinity_key(turn2) is not None


def test_affinity_key_differs_between_conversations_and_tool_sets():
    a = balancer_mod.affinity_key(chat(SYSTEM, {"role": "user", "content": "pod is pending"}))
    b = balancer_mod.affinity_key(chat(SYSTEM, {"role": "user", "content": "service has no endpoints"}))
    c = balancer_mod.affinity_key(chat(SYSTEM, {"role": "user", "content": "pod is pending"}, tools=[{"x": 1}]))
    assert len({a, b, c}) == 3


def test_affinity_key_needs_a_chat_or_prompt():
    assert balancer_mod.affinity_key(b"not json {") is None
    assert balancer_mod.affinity_key(b'{"input": "embed me"}') is None
    assert balancer_mod.affinity_key(b'{"messages": 3}') is None
    assert balancer_mod.affinity_key(b'{"prompt": "abc"}') == balancer_mod.affinity_key(b'{"prompt": "abc"}')


def test_pick_prefers_the_hashed_instance_until_the_load_differs_too_much():
    key = "1"  # odd: the second of two ports
    assert balancer_mod.pick({8090: 0, 8091: 0}, key, 4) == 8091
    assert balancer_mod.pick({8090: 0, 8091: 4}, key, 4) == 8091  # within the slack
    assert balancer_mod.pick({8090: 0, 8091: 5}, key, 4) == 8090  # the load wins
    assert balancer_mod.pick({8090: 0, 8091: 0}, key, -1) == 8090  # off: least busy (the first on a tie)
    assert balancer_mod.pick({8090: 3, 8091: 0}, None, 4) == 8091  # no key: least busy
    assert balancer_mod.pick({8090: 7}, key, 4) == 8090  # one instance


async def test_turns_of_one_conversation_reach_the_same_instance(aiohttp_client, balancer):
    client = await aiohttp_client(balancer.app())

    async def turn(*messages):
        resp = await client.post("/v1/chat/completions", data=chat(*messages), headers=AUTH)
        return (await resp.json())["backend"]

    history = [SYSTEM, {"role": "user", "content": "pod is pending"}]
    seen = set()
    for i in range(4):
        seen.add(await turn(*history))
        history += [{"role": "assistant", "content": f"step {i}"}, {"role": "tool", "content": "ok"}]
    assert len(seen) == 1
    others = {await turn(SYSTEM, {"role": "user", "content": f"task {i}"}) for i in range(8)}
    assert others == {"gpu0", "gpu1"}  # other conversations spread over both


async def test_sticky_routing_can_be_turned_off(aiohttp_client, balancer, server, monkeypatch):
    monkeypatch.setattr(server.SETTINGS, "sticky_slack", -1)
    client = await aiohttp_client(balancer.app())

    async def one():
        resp = await client.post(
            "/v1/chat/completions", data=chat(SYSTEM, {"role": "user", "content": "same"}), headers=AUTH
        )
        return (await resp.json())["backend"]

    assert sorted(await asyncio.gather(*(one() for _ in range(6)))) == ["gpu0"] * 3 + ["gpu1"] * 3
