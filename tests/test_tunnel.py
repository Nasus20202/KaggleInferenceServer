"""Detecting a dead tunnel: the server restarts cloudflared, the proxy gives up."""

import asyncio
import threading

import pytest
from aiohttp import web
from conftest import SECRETS

from kis import events, proxy


async def test_tunnel_alive_by_health_status(aiohttp_server, server):
    async def health(req: web.Request) -> web.Response:
        return web.json_response({"status": "ok"})

    async def edge_error(req: web.Request) -> web.Response:
        return web.Response(status=530)

    up = web.Application()
    up.router.add_get("/health", health)
    down = web.Application()
    down.router.add_get("/health", edge_error)
    up_url = str((await aiohttp_server(up)).make_url("")).rstrip("/")
    down_url = str((await aiohttp_server(down)).make_url("")).rstrip("/")
    assert await asyncio.to_thread(server.tunnel_alive, up_url)
    assert not await asyncio.to_thread(server.tunnel_alive, down_url)


def test_tunnel_alive_is_false_without_a_host(server):
    assert not server.tunnel_alive("http://127.0.0.1:1", timeout=1)


class FakeProc:
    def __init__(self) -> None:
        self.killed = threading.Event()

    def poll(self) -> int | None:
        return -9 if self.killed.is_set() else None

    def kill(self) -> None:
        self.killed.set()


def watch(server, monkeypatch, probes: list[bool]) -> FakeProc:
    """Run watch_tunnel through `probes`; a probe past the list ends the proc."""
    proc, results = FakeProc(), iter(probes)

    def alive(url: str, timeout: float = 15) -> bool:
        try:
            return next(results)
        except StopIteration:
            proc.killed.set()
            return True

    monkeypatch.setattr(server, "tunnel_alive", alive)
    server.watch_tunnel("https://t.example", proc, interval=0)
    return proc


def test_kills_cloudflared_after_consecutive_failures(server, monkeypatch):
    proc = watch(server, monkeypatch, [True, False, False, False, True])
    assert proc.killed.is_set()
    assert proc.poll() == -9


def test_a_recovered_probe_resets_the_count(server, monkeypatch):
    killed = []
    proc = FakeProc()
    proc.kill = lambda: killed.append(1)  # type: ignore[method-assign]
    results = iter([True, False, False, True, False, False])
    ended = []

    def alive(url: str, timeout: float = 15) -> bool:
        try:
            return next(results)
        except StopIteration:
            ended.append(1)
            proc.poll = lambda: 0  # type: ignore[method-assign]
            return True

    monkeypatch.setattr(server, "tunnel_alive", alive)
    server.watch_tunnel("https://t.example", proc, interval=0)  # type: ignore[arg-type]
    assert ended and not killed


def test_a_tunnel_that_never_worked_is_left_alone(server, monkeypatch):
    killed = []
    proc = FakeProc()
    proc.kill = lambda: killed.append(1)  # type: ignore[method-assign]
    results = iter([False] * 6)

    def alive(url: str, timeout: float = 15) -> bool:
        try:
            return next(results)
        except StopIteration:
            proc.poll = lambda: 0  # type: ignore[method-assign]
            return False

    monkeypatch.setattr(server, "tunnel_alive", alive)
    server.watch_tunnel("https://t.example", proc, interval=0)  # type: ignore[arg-type]
    assert not killed


async def dead_tunnel_client(aiohttp_client, aiohttp_server, monkeypatch, config, give_up, minutes="0.001"):
    async def gone(req: web.Request) -> web.Response:
        return web.Response(status=530)

    app = web.Application()
    app.router.add_route("*", "/{tail:.*}", gone)
    url = str((await aiohttp_server(app)).make_url("")).rstrip("/")
    monkeypatch.setenv("KIS_PROXY_GIVE_UP_MINUTES", minutes)
    monkeypatch.setattr(events, "endpoint", lambda config, topic: url)
    monkeypatch.setattr(proxy.retry, "DELAYS", (0.0,))
    return await aiohttp_client(proxy.make_app(config, SECRETS, give_up=give_up))


async def test_proxy_gives_up_on_a_dead_tunnel(aiohttp_client, aiohttp_server, monkeypatch, config):
    stopped = asyncio.Event()
    client = await dead_tunnel_client(aiohttp_client, aiohttp_server, monkeypatch, config, stopped.set)
    assert (await client.get("/v1/models")).status == 530
    await asyncio.wait_for(stopped.wait(), timeout=10)


async def test_proxy_keeps_running_without_the_setting(aiohttp_client, aiohttp_server, monkeypatch, config):
    stopped = asyncio.Event()
    client = await dead_tunnel_client(aiohttp_client, aiohttp_server, monkeypatch, config, stopped.set, minutes="")
    assert (await client.get("/v1/models")).status == 530
    await asyncio.sleep(0.2)
    assert not stopped.is_set()


def test_a_success_stops_the_clock():
    up = proxy.Upstream("http://x")
    up.reached(False)
    assert up.down_since is not None
    first = up.down_since
    up.reached(False)
    assert up.down_since == first
    up.reached(True)
    assert up.down_since is None


@pytest.fixture(autouse=True)
def _no_real_sleep_in_proxy(monkeypatch):
    monkeypatch.setattr(proxy.retry, "ATTEMPTS", 2)


async def test_admin_client_sends_its_own_user_agent(aiohttp_server):
    """urllib's default User-Agent is rejected by Cloudflare in front of a named tunnel."""
    from kis.client import USER_AGENT, AdminClient

    seen = []

    async def health(req: web.Request) -> web.Response:
        seen.append(req.headers.get("User-Agent"))
        return web.json_response({})

    app = web.Application()
    app.router.add_post("/admin/shutdown", health)
    url = str((await aiohttp_server(app)).make_url("")).rstrip("/")
    await asyncio.to_thread(AdminClient(url, "key").shutdown, "abc")
    assert seen == [USER_AGENT]
