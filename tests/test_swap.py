"""Model hotswap in the Kaggle-side balancer (kaggle/server.py): autoload, /admin/load, /v1/models."""

import asyncio

import pytest
from conftest import FakeBackends, fake_llama_server

from kis.schema import EventType, Model

AUTH = {"Authorization": "Bearer secret"}
PRESETS = {
    "a": Model(repo="r", file="a.gguf", alias="A-Q4", parallel=1),
    "b": Model(repo="r", file="b.gguf", alias="B-Q4", parallel=1),
}


@pytest.fixture
def events(server, monkeypatch):
    sent = []
    monkeypatch.setattr(server, "notify", lambda event, **data: sent.append((event, data)))
    monkeypatch.setattr(server, "RUNTIME", server.Runtime())
    return sent


@pytest.fixture
async def swapper(server, aiohttp_server, aiohttp_client, events):
    """A balancer with presets a (two instances) and b (one), a loaded; returns (balancer, backends, client)."""

    async def backend(name):
        return (await aiohttp_server(fake_llama_server(name, delay=0.2))).port

    backends = FakeBackends({"a": [await backend("a0"), await backend("a1")], "b": [await backend("b0")]}, "a")
    balancer = server.Balancer(backends, "secret", presets=PRESETS, loaded="a")
    return balancer, backends, await aiohttp_client(balancer.app())


async def chat(client, model=None):
    body = {} if model is None else {"model": model}
    resp = await client.post("/v1/chat/completions", json=body, headers=AUTH)
    return resp.status, await resp.json()


async def test_a_request_for_another_preset_swaps_to_it(swapper, events):
    balancer, backends, client = swapper
    status, body = await chat(client, "b")
    assert status == 200 and body["backend"] == "b0"
    assert backends.loads == ["b"] and balancer.loaded == "b"
    assert [e for e, _ in events] == [EventType.MODEL_LOADING, EventType.MODEL_READY]
    assert events[-1][1]["preset"] == "b" and events[-1][1]["model"] == "B-Q4"
    _, body = await chat(client)  # no model: whichever is loaded
    assert body["backend"] == "b0"


async def test_presets_are_named_by_preset_name_or_alias(swapper):
    balancer, backends, client = swapper
    await chat(client, "B-Q4")
    assert balancer.loaded == "b"
    await chat(client, "a")
    assert backends.loads == ["b", "a"]


async def test_unknown_model_names_do_not_swap(swapper):
    _, backends, client = swapper
    status, body = await chat(client, "gpt-4o")
    assert status == 200 and body["backend"].startswith("a")
    assert not backends.loads


async def test_models_lists_every_preset_with_its_status(swapper):
    _, _, client = swapper
    assert (await client.get("/v1/models")).status == 401
    models = await (await client.get("/v1/models", headers=AUTH)).json()
    assert [(m["id"], m["preset"], m["status"]) for m in models["data"]] == [
        ("A-Q4", "a", "loaded"),
        ("B-Q4", "b", "available"),
    ]
    await chat(client, "b")
    models = await (await client.get("/models", headers=AUTH)).json()
    assert [m["status"] for m in models["data"]] == ["downloaded", "loaded"]


async def test_the_loaded_preset_keeps_serving_while_the_next_downloads(swapper):
    balancer, backends, client = swapper
    backends.fetch_s = 0.6
    swap = asyncio.create_task(chat(client, "b"))
    await asyncio.sleep(0.1)
    statuses = await (await client.get("/v1/models", headers=AUTH)).json()
    assert [m["status"] for m in statuses["data"]] == ["loaded", "loading"]
    _, body = await chat(client)  # answered by a, before b has even downloaded
    assert body["backend"].startswith("a") and not backends.loads
    assert balancer.idle_seconds == 0
    assert (await swap)[1]["backend"] == "b0"


async def test_a_swap_waits_for_the_requests_in_flight(swapper, monkeypatch):
    balancer, backends, client = swapper
    in_flight_at_load = []
    load = backends.load
    monkeypatch.setattr(
        backends, "load", lambda name: in_flight_at_load.append(sum(balancer.inflight.values())) or load(name)
    )
    first = asyncio.create_task(chat(client))
    await asyncio.sleep(0.05)
    (status_a, body_a), (status_b, body_b) = await asyncio.gather(first, chat(client, "b"))
    assert (status_a, status_b) == (200, 200)
    assert body_a["backend"].startswith("a") and body_b["backend"] == "b0"
    assert in_flight_at_load == [0]


async def test_requests_for_the_old_preset_wait_during_a_swap(swapper):
    """A busy preset can't keep another from loading: once a swap drains, everyone waits for it."""
    _, backends, client = swapper
    results = await asyncio.gather(chat(client, "b"), *(chat(client, "a") for _ in range(3)))
    assert {status for status, _ in results} == {200}
    assert results[0][1]["backend"] == "b0"
    assert backends.loads[0] == "b"


async def test_a_failed_swap_reloads_the_previous_preset(swapper, events):
    balancer, backends, client = swapper
    backends.fail = ("b",)
    status, body = await chat(client, "b")
    assert status == 503 and "loading B-Q4 failed" in body["error"]["message"]
    assert backends.loads == ["b", "a"] and balancer.loaded == "a"
    assert EventType.MODEL_FAILED in [e for e, _ in events]
    status, body = await chat(client)
    assert status == 200 and body["backend"].startswith("a")
    assert not balancer.stop.is_set()


async def test_the_session_stops_when_the_previous_preset_fails_to_reload_too(swapper):
    balancer, backends, client = swapper
    backends.fail = ("a", "b")
    status, _ = await chat(client, "b")
    assert status == 503
    assert balancer.stop.is_set() and "after a failed swap" in balancer.failure


async def test_admin_load_swaps_in_the_background(swapper):
    balancer, backends, client = swapper
    assert (await client.post("/admin/load?model=b")).status == 401
    assert (await client.post("/admin/load?model=nope", headers=AUTH)).status == 404
    resp = await client.post("/admin/load?model=b", headers=AUTH)
    assert resp.status == 202 and (await resp.json())["status"] == "loading"
    for _ in range(50):
        if balancer.loaded == "b" and not balancer.swap:
            break
        await asyncio.sleep(0.02)
    resp = await client.post("/admin/load?model=B-Q4", headers=AUTH)
    assert resp.status == 200 and (await resp.json())["status"] == "loaded"
    assert backends.loads == ["b"]


async def test_without_autoload_only_admin_load_swaps(swapper, server, monkeypatch):
    _, backends, client = swapper
    monkeypatch.setattr(server.SETTINGS, "autoload", False)
    status, body = await chat(client, "b")
    assert status == 400 and body["error"]["code"] == "model_not_loaded"
    assert (await chat(client, "gpt-4o"))[0] == 200  # unknown names still go to the loaded preset
    assert (await client.post("/admin/load?model=b", headers=AUTH)).status == 202
    assert not backends.loads or backends.loads == ["b"]


async def test_keepalive_while_waiting_for_a_swap(swapper, server, monkeypatch):
    """A swap can take minutes: bytes flow before Cloudflare's 100 s limit, then the answer."""
    monkeypatch.setattr(server, "KEEPALIVE_SECONDS", 0.05)
    _, backends, client = swapper
    backends.fetch_s = 0.3
    resp = await client.post("/v1/chat/completions", json={"model": "b", "stream": True}, headers=AUTH)
    lines = [line async for line in resp.content if line.strip()]
    assert lines[0].startswith(b": keepalive") and any(b'"b0"' in line for line in lines)


async def test_a_failed_swap_after_keepalive_sends_the_error_in_the_body(swapper, server, monkeypatch):
    monkeypatch.setattr(server, "KEEPALIVE_SECONDS", 0.05)
    _, backends, client = swapper
    backends.fetch_s, backends.fail = 0.3, ("b",)
    resp = await client.post("/v1/chat/completions", json={"model": "b"}, headers=AUTH)
    assert resp.status == 200  # sent with the first keepalive byte
    assert "loading B-Q4 failed" in (await resp.json(content_type=None))["error"]["message"]


async def test_models_report_context_and_slots(swapper, server, monkeypatch):
    _, _, client = swapper
    monkeypatch.setattr(server, "RUNTIME", server.Runtime("a", "A-Q4", "replicas", slots=8, parallel=4, ctx=65536))
    monkeypatch.setattr(server.SETTINGS, "ctx", 32768)
    models = (await (await client.get("/v1/models", headers=AUTH)).json())["data"]
    layout = [(m["context_length"], m["parallel"], m["slots"], m["topology"]) for m in models]
    assert layout == [(65536, 4, 8, "replicas"), (32768, 1, None, None)]  # a as running, b as configured


@pytest.fixture
async def with_resident(server, aiohttp_server, aiohttp_client):
    """A balancer with preset a loaded and resident preset e; returns (balancer, backends, client, e's server)."""
    embedder = await aiohttp_server(fake_llama_server("e0"))
    presets = {**PRESETS, "e": Model(repo="r", file="e.gguf", alias="E-Q8", parallel=4, ctx=2048)}
    backends = FakeBackends({"a": [(await aiohttp_server(fake_llama_server("a0"))).port], "b": []}, "a")
    balancer = server.Balancer(backends, "secret", presets=presets, loaded="a", residents={"e": embedder.port})
    return balancer, backends, await aiohttp_client(balancer.app()), embedder


async def test_requests_for_a_resident_preset_go_to_it_and_never_swap(with_resident):
    balancer, backends, client, embedder = with_resident
    for name in ("e", "E-Q8"):
        resp = await client.post("/v1/embeddings", json={"model": name}, headers=AUTH)
        assert (await resp.json())["backend"] == "e0"
    assert (await chat(client, "A-Q4"))[1]["backend"] == "a0"
    assert len(embedder.app["received"]) == 2
    assert backends.loads == []
    assert balancer.resident_inflight == 0


async def test_resident_presets_are_listed_as_loaded(with_resident):
    _, backends, client, _ = with_resident
    models = {m["preset"]: m for m in (await (await client.get("/v1/models", headers=AUTH)).json())["data"]}
    assert (models["e"]["status"], models["e"]["resident"], models["e"]["slots"]) == ("loaded", True, 4)
    assert models["a"]["resident"] is False
    resp = await client.post("/admin/load?model=e", headers=AUTH)
    assert (await resp.json())["status"] == "loaded"
    assert backends.loads == []
