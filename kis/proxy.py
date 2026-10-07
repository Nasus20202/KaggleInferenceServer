"""Local OpenAI-compatible endpoint that forwards to the current Kaggle server.

Quick-tunnel URLs change every session; the proxy re-reads the endpoint from
the event topic, so local clients can keep http://127.0.0.1:<port>/v1.
"""

import asyncio
import hmac
import logging
import os
import time

import aiohttp
from aiohttp import web

from . import events, retry, rollover

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


class Retryable(Exception):
    """A transient upstream status (502 while the tunnel reconnects, 524, ...)."""


def run(config: dict, secrets: dict, host: str, port: int):
    web.run_app(make_app(config, secrets), host=host, port=port, print=lambda msg: log.info(msg.strip()))


def authorized(req: web.Request, key: str) -> bool:
    """OpenAI-style `Authorization: Bearer <key>` or `x-api-key: <key>`."""
    given = req.headers.get("x-api-key") or req.headers.get("Authorization", "").removeprefix("Bearer ")
    return hmac.compare_digest(given.encode(), key.encode())


def make_app(
    config: dict, secrets: dict, refresh_seconds: float = 15, rollover_check_seconds: float = 60
) -> web.Application:
    """Without KIS_PROXY_API_KEY any client may use the proxy (it listens on localhost by
    default); with it, clients must send that key. With server.rollover_hours set, the
    proxy replaces a session that old with a fresh one, without downtime."""
    local_key = os.environ.get("KIS_PROXY_API_KEY", "")
    if local_key:
        log.info("clients must send KIS_PROXY_API_KEY")
    upstream = {"url": events.endpoint(config, secrets["ntfy_topic"])}
    log.info("upstream: %s", upstream["url"] or "none yet (kis up <model>)")

    async def follow_endpoint():
        while True:
            await asyncio.sleep(refresh_seconds)
            url = await asyncio.to_thread(events.endpoint, config, secrets["ntfy_topic"])
            if url != upstream["url"]:
                log.info("upstream: %s", url or "none (server stopped)")
                upstream["url"] = url

    def idle_seconds(session: dict) -> float:
        try:
            code, stats = rollover.admin(session["endpoint"], secrets["api_key"], "/admin/stats")
        except OSError:
            return 0  # unknown: assume busy
        return stats.get("idle_s", 0) if code == 200 else 0

    async def auto_rollover(hours: float):
        topic = secrets["ntfy_topic"]
        while True:
            await asyncio.sleep(rollover_check_seconds)
            current = await asyncio.to_thread(events.current, config, topic)
            if not current or time.time() - current["started"] < hours * 3600:
                continue
            if await asyncio.to_thread(idle_seconds, current) > ROLLOVER_MAX_IDLE_S:
                continue  # unused: let it stop on its idle timeout instead of paying for a replacement
            log.info("session %s is %.1f h old: starting its replacement", current["session"], hours)
            try:
                old, url = await asyncio.to_thread(rollover.rollover, config, topic)
                upstream["url"] = url  # new requests go to the new session from now on
                log.info("upstream: %s (rollover); draining session %s", url, old["session"])
                await asyncio.to_thread(rollover.retire, old["endpoint"], secrets["api_key"], old["session"])
            except Exception as e:
                log.error("rollover failed: %s; next try in 10 min", e)
                await asyncio.sleep(600)

    async def handle(req: web.Request) -> web.StreamResponse:
        if local_key and not authorized(req, local_key):
            log.warning("%s %s 401 (wrong or missing KIS_PROXY_API_KEY)", req.method, req.path)
            return web.json_response({"error": {"message": "invalid api key"}}, status=401)
        if not upstream["url"]:
            return web.json_response({"error": {"message": "no Kaggle server running (kis up <model>)"}}, status=503)
        headers = {k: v for k, v in req.headers.items() if k.lower() not in HOP_HEADERS}
        headers["Authorization"] = f"Bearer {secrets['api_key']}"
        start, status = time.time(), "-"
        try:
            for attempt in range(retry.ATTEMPTS):
                last = attempt == retry.ATTEMPTS - 1
                try:
                    resp = await forward(req, upstream["url"] + str(req.rel_url), headers, retry_status=not last)
                    status = resp.status
                    return resp
                except (aiohttp.ClientConnectionError, TimeoutError, Retryable) as e:
                    if last:
                        log.warning("Kaggle server unreachable: %s", e)
                        status = 502
                        return web.json_response({"error": {"message": f"Kaggle server unreachable: {e}"}}, status=502)
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
                    upstream["url"] = (
                        await asyncio.to_thread(events.endpoint, config, secrets["ntfy_topic"]) or upstream["url"]
                    )
            raise AssertionError("unreachable")
        finally:
            log.info("%s %s %s %.2fs", req.method, req.path, status, time.time() - start)

    async def forward(req: web.Request, url: str, headers: dict, retry_status: bool) -> web.StreamResponse:
        """Stream one upstream response. Errors before the response starts are raised (and
        retried); once bytes reach the client, a broken upstream just ends the response."""
        async with req.app["session"].request(req.method, url, data=await req.read(), headers=headers) as up:
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

    async def lifecycle(app: web.Application):
        app["session"] = aiohttp.ClientSession(timeout=aiohttp.ClientTimeout(total=None, sock_connect=20))
        tasks = [asyncio.create_task(follow_endpoint())]
        if hours := config["server"].get("rollover_hours"):
            log.info("rollover after %.1f h per session", hours)
            tasks.append(asyncio.create_task(auto_rollover(hours)))
        yield
        for task in tasks:
            task.cancel()
        await app["session"].close()

    app = web.Application(client_max_size=256 << 20)
    app.cleanup_ctx.append(lifecycle)
    app.router.add_route("*", "/{tail:.*}", handle)
    return app
