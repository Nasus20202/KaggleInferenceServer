"""Session rollover (kis/rollover.py) and the proxy switching sessions without downtime."""

import asyncio
import dataclasses
from http import HTTPStatus

import pytest
from conftest import SECRETS, FakeBackends, fake_llama_server, make_balancer

from kis import events, proxy, rollover, settings
from kis.client import AdminClient
from kis.schema import Event, EventType, Model, ServerConfig
from kis.status import Quota

SERVER = ServerConfig(api_key="k", model=Model(repo="r", file="f", alias="m", parallel=1), max_hours=11.5)
OLD = events.Session("old", endpoint="https://old")


@pytest.fixture(autouse=True)
def state_dir(monkeypatch, tmp_path):
    monkeypatch.setattr(settings, "STATE_DIR", tmp_path)


def test_max_hours_end_before_the_quota_runs_out():
    assert rollover.capped_max_hours(11.5, None) == 11.5
    assert rollover.capped_max_hours(11.5, Quota(used_h=1.0, total_h=30.0)) == 11.5
    assert rollover.capped_max_hours(11.5, Quota(used_h=28.0, total_h=30.0)) == pytest.approx(2 - 2 / 60, abs=1e-3)
    assert rollover.capped_max_hours(11.5, Quota(used_h=30.5, total_h=30.0)) == 0


def test_launch_pushes_a_new_session_and_remembers_it(monkeypatch, config):
    pushed = []
    monkeypatch.setattr(rollover.kaggle, "push", lambda config, slug, script, params, sources: pushed.append(params))
    monkeypatch.setattr(rollover.status, "quota", lambda: Quota(used_h=29.0, total_h=30.0))
    session = rollover.launch(config, SERVER, "kis-server-b")
    sent = pushed[0]["CONFIG"]
    assert sent.session == session and sent.max_hours < 1
    assert settings.load_state() == settings.State(server=SERVER, slug="kis-server-b", session=session, model="m")


def test_wait_ready_follows_only_its_session(monkeypatch, config):
    feed = [
        ("1", Event(EventType.READY, session="old", data={"endpoint": "https://old"})),
        ("2", Event(EventType.STARTING, session="new")),
        ("3", Event(EventType.READY, session="new", data={"endpoint": "https://new"})),
    ]
    monkeypatch.setattr(events, "fetch", lambda ntfy, topic, since: feed)
    assert rollover.wait_ready(config, "t", "new", "0") == "https://new"
    feed[2] = ("3", Event(EventType.ERROR, session="new", data={"error": "OOM"}))
    with pytest.raises(rollover.RolloverError, match="OOM"):
        rollover.wait_ready(config, "t", "new", "0")


def test_rollover_alternates_kernel_slugs(monkeypatch, config):
    monkeypatch.setattr(events, "current", lambda config, topic: OLD)
    monkeypatch.setattr(rollover.status, "quota", lambda: None)
    slugs = []
    monkeypatch.setattr(rollover, "launch", lambda config, server, slug: slugs.append(slug) or "new")
    monkeypatch.setattr(rollover, "wait_ready", lambda config, topic, session, since: "https://new")
    settings.save_state(settings.State(server=SERVER, slug="kis-server"))
    assert rollover.rollover(config, "t") == (OLD, "https://new")
    settings.save_state(settings.State(server=SERVER, slug="kis-server-b"))
    rollover.rollover(config, "t")
    assert slugs == ["kis-server-b", "kis-server"]


def test_rollover_starts_with_the_loaded_preset(monkeypatch, config):
    other = Model(repo="r", file="g", alias="g", parallel=2)
    server = dataclasses.replace(SERVER, preset="m", models={"m": SERVER.model, "g": other})
    monkeypatch.setattr(events, "current", lambda config, topic: dataclasses.replace(OLD, preset="g"))
    monkeypatch.setattr(rollover.status, "quota", lambda: None)
    launched = []
    monkeypatch.setattr(rollover, "launch", lambda config, server, slug: launched.append(server) or "new")
    monkeypatch.setattr(rollover, "wait_ready", lambda config, topic, session, since: "https://new")
    settings.save_state(settings.State(server=server))
    rollover.rollover(config, "t")
    assert (launched[0].preset, launched[0].model) == ("g", other)


def test_rollover_refuses_without_quota(monkeypatch, config):
    monkeypatch.setattr(events, "current", lambda config, topic: OLD)
    monkeypatch.setattr(rollover.status, "quota", lambda: Quota(used_h=29.9, total_h=30.0))
    settings.save_state(settings.State(server=SERVER))
    with pytest.raises(rollover.RolloverError, match="quota"):
        rollover.rollover(config, "t")


def test_rollover_needs_the_settings_of_kis_up(monkeypatch, config):
    monkeypatch.setattr(events, "current", lambda config, topic: OLD)
    with pytest.raises(rollover.RolloverError, match="kis up"):
        rollover.rollover(config, "t")


async def balancer_server(server, aiohttp_server, name: str, delay: float):
    backend = await aiohttp_server(fake_llama_server(name, delay=delay))
    balancer = make_balancer(server, FakeBackends({server.START: [backend.port]}, server.START))
    site = await aiohttp_server(balancer.app())
    return balancer, str(site.make_url("")).rstrip("/")


async def test_retire_waits_for_in_flight_requests(server, aiohttp_server, aiohttp_client, monkeypatch):
    monkeypatch.setattr(server.SETTINGS, "session", "old")
    balancer, url = await balancer_server(server, aiohttp_server, "old", delay=1.0)
    client = await aiohttp_client(balancer.app())
    slow = asyncio.create_task(client.post("/v1/chat/completions", json={}, headers={"Authorization": "Bearer secret"}))
    await asyncio.sleep(0.2)
    await asyncio.to_thread(rollover.retire, url, "secret", "old", 30, 0.1, grace=0)
    assert balancer.stop.is_set()
    assert (await slow).status == 200  # finished before the shutdown


async def test_retire_leaves_another_session_running(server, aiohttp_server, monkeypatch):
    monkeypatch.setattr(server.SETTINGS, "session", "new")
    balancer, url = await balancer_server(server, aiohttp_server, "new", delay=0)
    await asyncio.to_thread(rollover.retire, url, "secret", "old", 0.3, 0.1, 0.5, grace=0)
    assert not balancer.stop.is_set()
    assert await asyncio.to_thread(AdminClient(url, "secret").shutdown, "old") == HTTPStatus.CONFLICT


async def test_admin_client_loads_a_preset(server, aiohttp_server, monkeypatch):
    monkeypatch.setattr(server.SETTINGS, "session", "s")
    _, url = await balancer_server(server, aiohttp_server, "s", delay=0)
    client = AdminClient(url, "secret")
    code, answer = await asyncio.to_thread(client.load, server.START)
    assert code == HTTPStatus.OK and answer["status"] == "loaded" and answer["session"] == "s"
    models = await asyncio.to_thread(client.models)
    assert [(m["preset"], m["status"]) for m in models] == [(server.START, "loaded")]
    code, answer = await asyncio.to_thread(client.load, "nope")
    assert code == HTTPStatus.NOT_FOUND and "available" in answer["error"]["message"]


async def test_proxy_switches_sessions_without_failing_a_request(
    server, aiohttp_server, aiohttp_client, monkeypatch, config
):
    """Requests keep flowing through the proxy while it replaces the old session."""
    monkeypatch.setattr(server.SETTINGS, "session", "old")
    old_balancer, old_url = await balancer_server(server, aiohttp_server, "old", delay=0.05)
    _, new_url = await balancer_server(server, aiohttp_server, "new", delay=0.05)
    sessions = {"current": events.Session("old", endpoint=old_url, started=0)}
    monkeypatch.setattr(events, "current", lambda config, topic: sessions["current"])
    monkeypatch.setattr(events, "endpoint", lambda config, topic: sessions["current"].endpoint)

    def fake_rollover(config, topic):
        import time

        time.sleep(0.5)  # the new session starts while the old one serves
        old = sessions["current"]
        sessions["current"] = events.Session("new", endpoint=new_url, started=1e12)
        return old, new_url

    monkeypatch.setattr(rollover, "rollover", fake_rollover)
    config.server = dataclasses.replace(config.server, rollover_hours=11.0)
    monkeypatch.setattr(rollover, "GRACE_S", 0)
    app = proxy.make_app(config, SECRETS, refresh_seconds=0.1, rollover_check_seconds=0.1)
    client = await aiohttp_client(app)

    backends, statuses = [], []
    for _ in range(40):
        resp = await client.post("/v1/chat/completions", json={})
        statuses.append(resp.status)
        backends.append((await resp.json())["backend"])
        await asyncio.sleep(0.03)
    assert set(statuses) == {200}
    assert backends[0] == "old" and backends[-1] == "new"
    for _ in range(50):
        if old_balancer.stop.is_set():
            break
        await asyncio.sleep(0.1)
    assert old_balancer.stop.is_set()  # the old session was shut down after draining


async def test_proxy_does_not_replace_an_idle_session(server, aiohttp_server, aiohttp_client, monkeypatch, config):
    monkeypatch.setattr(server.SETTINGS, "session", "old")
    balancer, url = await balancer_server(server, aiohttp_server, "old", delay=0)
    balancer.last_activity -= 3600  # idle for an hour
    monkeypatch.setattr(events, "current", lambda config, topic: events.Session("old", endpoint=url, started=0))
    monkeypatch.setattr(events, "endpoint", lambda config, topic: url)
    started = []
    monkeypatch.setattr(rollover, "rollover", lambda config, topic: started.append(1))
    config.server = dataclasses.replace(config.server, rollover_hours=11.0)
    await aiohttp_client(proxy.make_app(config, SECRETS, refresh_seconds=0.1, rollover_check_seconds=0.05))
    await asyncio.sleep(0.5)
    assert not started and not balancer.stop.is_set()
