"""Local OpenAI-compatible endpoint that forwards to the current Kaggle server.

Quick-tunnel URLs change every session; the proxy re-reads the endpoint from
the event topic, so local clients can keep http://127.0.0.1:<port>/v1.
"""

import asyncio

import aiohttp
from aiohttp import web

from . import events

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


def run(config: dict, secrets: dict, host: str, port: int):
    web.run_app(make_app(config, secrets), host=host, port=port)


def make_app(config: dict, secrets: dict, refresh_seconds: float = 15) -> web.Application:
    upstream = {"url": events.endpoint(config, secrets["ntfy_topic"])}
    print(f"upstream: {upstream['url'] or 'none yet'}")

    async def follow_endpoint():
        while True:
            await asyncio.sleep(refresh_seconds)
            url = await asyncio.to_thread(events.endpoint, config, secrets["ntfy_topic"])
            if url != upstream["url"]:
                print(f"upstream: {url}")
                upstream["url"] = url

    async def handle(req: web.Request) -> web.StreamResponse:
        if not upstream["url"]:
            return web.json_response({"error": {"message": "no Kaggle server running (kis up <model>)"}}, status=503)
        headers = {k: v for k, v in req.headers.items() if k.lower() not in HOP_HEADERS}
        headers["Authorization"] = f"Bearer {secrets['api_key']}"
        url = upstream["url"] + str(req.rel_url)
        try:
            return await forward(req, url, headers)
        except aiohttp.ClientConnectionError as e:
            return web.json_response({"error": {"message": f"Kaggle server unreachable: {e}"}}, status=502)

    async def forward(req: web.Request, url: str, headers: dict) -> web.StreamResponse:
        async with req.app["session"].request(req.method, url, data=await req.read(), headers=headers) as up:
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
            return resp

    async def lifecycle(app: web.Application):
        app["session"] = aiohttp.ClientSession(timeout=aiohttp.ClientTimeout(total=None, sock_connect=20))
        task = asyncio.create_task(follow_endpoint())
        yield
        task.cancel()
        await app["session"].close()

    app = web.Application(client_max_size=256 << 20)
    app.cleanup_ctx.append(lifecycle)
    app.router.add_route("*", "/{tail:.*}", handle)
    return app
