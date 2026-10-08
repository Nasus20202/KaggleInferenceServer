"""Local OpenAI-compatible endpoint that forwards to the current Kaggle server.

Quick-tunnel URLs change every session; the proxy re-reads the endpoint from
the event topic, so local clients can keep http://127.0.0.1:<port>/v1.
"""

import asyncio
import hmac
import logging
import os
import signal
import time
from collections.abc import AsyncIterator, Callable
from dataclasses import dataclass

import aiohttp
from aiohttp import web

from . import events, retry, rollover
from .client import AdminClient
from .schema import api_error
from .settings import Config, Secrets

log = logging.getLogger("kis.proxy")

HOP_HEADERS = {
    "host",
    "authorization",
    "x-api-key",
    "content-length",
    "transfer-encoding",
    "connection",
    "accept-encoding",
    "content-encoding",
}


ROLLOVER_MAX_IDLE_S = 300  # an older session idle for longer isn't replaced
CLIENT = web.AppKey("client", aiohttp.ClientSession)


class Retryable(Exception):
    """A transient upstream status (502 while the tunnel reconnects, 524, ...)."""


TUNNEL_DOWN = 530  # Cloudflare: no tunnel is registered for the hostname


@dataclass
class Upstream:
    url: str | None  # base URL of the current Kaggle server
    down_since: float | None = None  # when requests started failing with no success since

    def reached(self, ok: bool) -> None:
        """Record one request's outcome: the first failure starts the clock, a success stops it."""
        if ok:
            self.down_since = None
        elif self.down_since is None:
            self.down_since = time.time()


def stop_process() -> None:
    os.kill(os.getpid(), signal.SIGTERM)


def run(config: Config, secrets: Secrets, host: str, port: int) -> None:
    web.run_app(make_app(config, secrets), host=host, port=port, print=lambda msg: log.info(msg.strip()))


def authorized(req: web.Request, key: str) -> bool:
    """OpenAI-style `Authorization: Bearer <key>` or `x-api-key: <key>`."""
    given = req.headers.get("x-api-key") or req.headers.get("Authorization", "").removeprefix("Bearer ")
    return hmac.compare_digest(given.encode(), key.encode())


def make_app(
    config: Config,
    secrets: Secrets,
    refresh_seconds: float = 15,
    rollover_check_seconds: float = 60,
    give_up: Callable[[], None] = stop_process,
) -> web.Application:
    """Without KIS_PROXY_API_KEY any client may use the proxy (it listens on localhost by
    default); with it, clients must send that key. With server.rollover_hours set, the
    proxy replaces a session that old with a fresh one, without downtime. With
    KIS_PROXY_GIVE_UP_MINUTES set, it stops (calling `give_up`) once the server has been
    unreachable for that long, so that clients fail fast instead of retrying a dead tunnel."""
    local_key = os.environ.get("KIS_PROXY_API_KEY", "")
    give_up_s = float(os.environ.get("KIS_PROXY_GIVE_UP_MINUTES") or 0) * 60
    if local_key:
        log.info("clients must send KIS_PROXY_API_KEY")
    topic = secrets.ntfy_topic
    upstream = Upstream(events.endpoint(config, topic))
    log.info("upstream: %s", upstream.url or "none yet (kis up <model>)")

    async def follow_endpoint() -> None:
        while True:
            await asyncio.sleep(refresh_seconds)
            url = await asyncio.to_thread(events.endpoint, config, topic)
            if url != upstream.url:
                log.info("upstream: %s", url or "none (server stopped)")
                upstream.url = url

    async def watch_upstream() -> None:
        while True:
            await asyncio.sleep(min(30, give_up_s / 4))
            if upstream.down_since is not None and time.time() - upstream.down_since >= give_up_s:
                log.error("Kaggle server unreachable for %.0f min: stopping the proxy", give_up_s / 60)
                give_up()
                return

    def idle_seconds(session: events.Session) -> float:
        if not session.endpoint:
            return 0
        try:
            stats = AdminClient(session.endpoint, secrets.api_key).stats()
        except OSError:
            return 0  # unknown: assume busy
        return stats.idle_s if stats else 0

    async def auto_rollover(hours: float) -> None:
        while True:
            await asyncio.sleep(rollover_check_seconds)
            current = await asyncio.to_thread(events.current, config, topic)
            if not current or time.time() - current.started < hours * 3600:
                continue
            if await asyncio.to_thread(idle_seconds, current) > ROLLOVER_MAX_IDLE_S:
                continue  # unused: let it stop on its idle timeout instead of paying for a replacement
            log.info("session %s is %.1f h old: starting its replacement", current.session, hours)
            try:
                old, url = await asyncio.to_thread(rollover.rollover, config, topic)
                upstream.url = url  # new requests go to the new session from now on
                log.info("upstream: %s (rollover); draining session %s", url, old.session)
                if old.endpoint:
                    await asyncio.to_thread(rollover.retire, old.endpoint, secrets.api_key, old.session)
            except Exception as e:
                log.error("rollover failed: %s; next try in 10 min", e)
                await asyncio.sleep(600)

    async def handle(req: web.Request) -> web.StreamResponse:
        if local_key and not authorized(req, local_key):
            log.warning("%s %s 401 (wrong or missing KIS_PROXY_API_KEY)", req.method, req.path)
            return web.json_response(api_error("invalid api key"), status=401)
        if not upstream.url:
            return web.json_response(api_error("no Kaggle server running (kis up <model>)"), status=503)
        headers = {k: v for k, v in req.headers.items() if k.lower() not in HOP_HEADERS}
        headers["Authorization"] = f"Bearer {secrets.api_key}"
        start, status = time.time(), 0
        try:
            for attempt in range(retry.ATTEMPTS):
                last = attempt == retry.ATTEMPTS - 1
                try:
                    resp = await forward(req, f"{upstream.url}{req.rel_url}", headers, retry_status=not last)
                    status = resp.status
                    upstream.reached(status != TUNNEL_DOWN)
                    return resp
                except (aiohttp.ClientConnectionError, TimeoutError, Retryable) as e:
                    if last:
                        log.warning("Kaggle server unreachable: %s", e)
                        upstream.reached(False)
                        status = 502
                        return web.json_response(api_error(f"Kaggle server unreachable: {e}"), status=502)
                    delay = retry.backoff(attempt)
                    log.warning(
                        "%s %s: %s; retry %d/%d in %.1fs",
                        req.method,
                        req.path,
                        e,
                        attempt + 1,
                        retry.ATTEMPTS - 1,
                        delay,
                    )
                    await asyncio.sleep(delay)
                    # the tunnel may have restarted with a new URL
                    upstream.url = await asyncio.to_thread(events.endpoint, config, topic) or upstream.url
            raise AssertionError("unreachable")
        finally:
            log.info("%s %s %s %.2fs", req.method, req.path, status or "-", time.time() - start)

    async def forward(req: web.Request, url: str, headers: dict[str, str], retry_status: bool) -> web.StreamResponse:
        """Stream one upstream response. Errors before the response starts are raised (and
        retried); once bytes reach the client, a broken upstream just ends the response."""
        async with req.app[CLIENT].request(req.method, url, data=await req.read(), headers=headers) as up:
            if retry_status and up.status in retry.RETRY_STATUS:
                raise Retryable(f"HTTP {up.status}")
            resp = web.StreamResponse(
                status=up.status, headers={k: v for k, v in up.headers.items() if k.lower() not in HOP_HEADERS}
            )
            await resp.prepare(req)
            try:
                async for chunk in up.content.iter_any():
                    await resp.write(chunk)
                await resp.write_eof()
            except ConnectionResetError:  # client went away mid-stream; closing upstream stops generation
                pass
            except (aiohttp.ClientError, TimeoutError) as e:
                log.warning("%s %s: upstream broke mid-response: %s", req.method, req.path, e)
            return resp

    async def lifecycle(app: web.Application) -> AsyncIterator[None]:
        app[CLIENT] = aiohttp.ClientSession(timeout=aiohttp.ClientTimeout(total=None, sock_connect=20))
        tasks = [asyncio.create_task(follow_endpoint())]
        if give_up_s:
            log.info("stopping after %.0f min without a reachable server", give_up_s / 60)
            tasks.append(asyncio.create_task(watch_upstream()))
        if hours := config.server.rollover_hours:
            log.info("rollover after %.1f h per session", hours)
            tasks.append(asyncio.create_task(auto_rollover(hours)))
        yield
        for task in tasks:
            task.cancel()
        await app[CLIENT].close()

    app = web.Application(client_max_size=256 << 20)
    app.cleanup_ctx.append(lifecycle)
    app.router.add_route("*", "/{tail:.*}", handle)
    return app
