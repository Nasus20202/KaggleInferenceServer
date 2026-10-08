"""The balancer: checks the API key, swaps presets, sends each request to the least busy
instance of the loaded preset and serves the /admin routes.

Inlined into server.py by `kis` (see state.py). The session's settings, the presets, the
runtime layout and `notify` come from server.py as arguments.
"""

import asyncio
import contextlib
import dataclasses
import json
import re
import time
from collections.abc import AsyncIterator, Awaitable, Callable
from dataclasses import dataclass
from typing import Any, Protocol

import aiohttp
from aiohttp import web

from backends import FITTED, BackendsFailed  # inlined by `kis`
from common import cpu_stats, gpu_stats, ram_stats  # inlined by `kis`
from kis.schema import (  # inlined by `kis`
    SERVER_LOG,
    EventType,
    Model,
    ModelStatus,
    Route,
    ServerConfig,
    Stats,
    UsageSummary,
    api_error,
    dump,
)
from state import PORT, STARTED, WORK, Runtime, log  # inlined by `kis`

KEEPALIVE_SECONDS = 30  # below Cloudflare's 100 s limit for the first response byte
DRAIN_POLL_S = 0.1  # how often a swap checks whether the requests in flight have finished


class BackendSet(Protocol):
    """What the balancer needs of the backends (Backends, or a stand-in in the tests)."""

    @property
    def ports(self) -> list[int]: ...
    @property
    def pids(self) -> dict[int, int]: ...
    def fetch(self, name: str) -> None: ...
    def downloaded(self, name: str) -> bool: ...
    def load(self, name: str) -> list[int]: ...


HOP_HEADERS = {
    "host",
    "authorization",
    "content-length",
    "transfer-encoding",
    "connection",
    "accept-encoding",
    "content-encoding",
}


CAPTURE_HEAD, CAPTURE_TAIL = 16 << 10, 256 << 10  # start and end of each response kept to read its usage


@dataclass
class Completion:
    """Tokens and timings of one completion (or, in Usage, the sum of many)."""

    model: str | None = None
    tokens_in: int = 0
    tokens_out: int = 0
    cached: int = 0
    prefill_n: int = 0  # prompt tokens processed (tokens_in - cached)
    prefill_ms: float = 0.0
    decode_ms: float = 0.0
    draft_n: int = 0
    draft_accepted: int = 0

    def describe(self) -> str:
        """One log fragment: tokens and speeds."""
        text = f"in={self.tokens_in} cached={self.cached} out={self.tokens_out}"
        if self.prefill_ms:
            text += f" prefill={self.prefill_n / self.prefill_ms * 1000:.0f}tok/s"
        if self.decode_ms:
            text += f" decode={self.tokens_out / self.decode_ms * 1000:.1f}tok/s"
        if self.draft_n:
            text += f" draft={self.draft_accepted / self.draft_n:.0%}"
        return text


def parse_usage(body: bytes, sse: bool) -> Completion | None:
    """Tokens and timings of one completion from llama-server's `usage` and `timings`
    (in the JSON body, or in the last events of an SSE stream); None if absent."""
    docs: list[object] = []
    if sse:
        for line in reversed(body.splitlines()):
            if line.startswith(b"data: {"):
                with contextlib.suppress(ValueError):
                    docs.append(json.loads(line[6:]))
            if len(docs) >= 3:
                break
    else:
        try:
            docs.append(json.loads(body))
        except ValueError:  # too large to keep whole, e.g. a batch of embeddings
            docs.append(kept_fields(body))
    dicts = [d for d in docs if isinstance(d, dict)]
    usage = next((d["usage"] for d in dicts if d.get("usage")), None)
    timings = next((d["timings"] for d in dicts if d.get("timings")), None)
    if not usage and not timings:
        return None
    usage, timings = usage or {}, timings or {}
    cached = (usage.get("prompt_tokens_details") or {}).get("cached_tokens", timings.get("cache_n", 0))
    return Completion(
        model=next((d["model"] for d in dicts if d.get("model")), None),
        tokens_in=usage.get("prompt_tokens", timings.get("prompt_n", 0) + timings.get("cache_n", 0)),
        tokens_out=usage.get("completion_tokens", timings.get("predicted_n", 0)),
        cached=cached or 0,
        prefill_n=timings.get("prompt_n", 0),
        prefill_ms=timings.get("prompt_ms", 0.0),
        decode_ms=timings.get("predicted_ms", 0.0),
        draft_n=timings.get("draft_n", 0),
        draft_accepted=timings.get("draft_n_accepted", 0),
    )


USAGE_KEY = re.compile(r'"(model|usage|timings)"\s*:\s*')


def kept_fields(body: bytes) -> dict[str, Any]:
    """`model`, `usage` and `timings` from the start and end kept of a JSON response whose
    middle was dropped: llama-server writes them before or after the large fields."""
    text, decoder, fields = body.decode(errors="replace"), json.JSONDecoder(), {}
    for match in USAGE_KEY.finditer(text):
        with contextlib.suppress(ValueError):
            fields.setdefault(match.group(1), decoder.raw_decode(text, match.end())[0])
    return fields


class Usage:
    """Running totals of completions for one model."""

    SUMMED = tuple(f.name for f in dataclasses.fields(Completion) if f.name != "model")

    def __init__(self) -> None:
        self.requests = 0
        self.totals = Completion()

    def add(self, c: Completion) -> None:
        self.requests += 1
        for name in self.SUMMED:
            setattr(self.totals, name, getattr(self.totals, name) + getattr(c, name))

    def summary(self) -> UsageSummary:
        t = self.totals
        return UsageSummary(
            requests=self.requests,
            tokens_in=t.tokens_in,
            tokens_out=t.tokens_out,
            tokens_cached=t.cached,
            cache_rate=round(t.cached / t.tokens_in, 3) if t.tokens_in else None,
            prefill_tps=round(t.prefill_n / t.prefill_ms * 1000, 1) if t.prefill_ms else None,
            decode_tps=round(t.tokens_out / t.decode_ms * 1000, 1) if t.decode_ms else None,
            draft_acceptance=round(t.draft_accepted / t.draft_n, 3) if t.draft_n else None,
        )


MODEL_ROUTES = (Route.MODELS, "/models")


class SwapFailed(RuntimeError):
    """The requested preset can't be served: loading it failed, or autoload is off."""


@dataclass
class Swap:
    """A swap in progress: first downloading (the loaded preset keeps serving), then
    draining the requests in flight and loading."""

    target: str
    task: "asyncio.Task[str | None] | None" = None  # result: an error, or None
    downloading: bool = True


class Balancer:
    """Checks the API key, swaps to the preset a request names (autoload) and sends each
    request to the instance of the loaded preset with the fewest in flight. Requests for
    a resident preset go to its instance."""

    def __init__(
        self,
        backends: BackendSet,
        api_key: str,
        settings: ServerConfig,
        runtime: Runtime,
        notify: Callable[..., None],
        presets: dict[str, Model],
        loaded: str,
        residents: dict[str, int] | None = None,
    ) -> None:
        self.backends = backends
        self.settings, self.runtime, self.notify = settings, runtime, notify
        self.residents = residents or {}  # resident preset -> port
        self.resident_inflight = 0
        self.presets = presets
        self.loaded = loaded  # preset name
        self.inflight = {port: 0 for port in backends.ports}
        self.swap: Swap | None = None
        self.waiting = 0  # requests waiting for a swap
        self.failure = ""  # why the session must stop: a failed swap left no preset loaded
        self.loads: set[asyncio.Task[None]] = set()  # swaps started by /admin/load
        self.api_key = api_key
        self.last_activity = time.time()
        self.stop = asyncio.Event()
        self.requests = self.errors = 0
        self.usage: dict[str, Usage] = {}
        self.session: aiohttp.ClientSession | None = None
        self.runner: web.AppRunner | None = None

    @property
    def idle_seconds(self) -> float:
        busy = self.swap or self.waiting or self.resident_inflight or any(self.inflight.values())
        return 0 if busy else time.time() - self.last_activity

    @property
    def loading(self) -> bool:
        """True while a swap has stopped (or is stopping) the loaded preset's instances."""
        return self.swap is not None and not self.swap.downloading

    def app(self) -> web.Application:
        async def client_session(app: web.Application) -> AsyncIterator[None]:
            self.session = aiohttp.ClientSession(timeout=aiohttp.ClientTimeout(total=None))
            yield
            await self.session.close()

        app = web.Application(client_max_size=256 << 20)
        app.cleanup_ctx.append(client_session)
        app.router.add_route("*", "/{tail:.*}", self.handle)
        return app

    async def start(self) -> None:
        self.runner = web.AppRunner(self.app())
        await self.runner.setup()
        await web.TCPSite(self.runner, "127.0.0.1", PORT).start()

    async def close(self) -> None:
        if self.runner:
            await self.runner.cleanup()

    async def handle(self, req: web.Request) -> web.StreamResponse:
        if req.path == Route.HEALTH:
            return web.json_response({"status": "ok"})
        if req.headers.get("Authorization") != f"Bearer {self.api_key}":
            return web.json_response(api_error("invalid api key"), status=401)
        if req.path == Route.SHUTDOWN:
            # With a named tunnel, two sessions share one URL during a rollover: only stop the one asked for.
            if req.query.get("session", self.settings.session) != self.settings.session:
                return web.json_response({**api_error("other session"), "session": self.settings.session}, status=409)
            log.info("shutdown requested")
            self.stop.set()
            return web.json_response({"status": "stopping", "session": self.settings.session})
        if req.path == Route.STATS:
            return web.json_response(dump(await self.stats()))
        if req.path == Route.LOGS:
            return self.logs(req)
        if req.path == Route.LOAD and req.method == "POST":
            return self.load(req.query.get("model", ""))
        if req.path in MODEL_ROUTES and req.method == "GET":
            return web.json_response(self.models())
        if self.full():
            log.warning("%s %s 429 (all slots busy, queue full)", req.method, req.path)
            return web.json_response(
                api_error("server busy: all slots in use and the queue is full", type="rate_limit"),
                status=429,
                headers={"Retry-After": "5"},
            )
        self.waiting += 1  # counted at once, so concurrent requests see each other in full()
        admitted = False
        self.requests += 1
        self.last_activity = start = time.time()
        port, status, detail = 0, 0, ""
        name: str | None = None

        async def connect() -> aiohttp.ClientResponse:
            nonlocal port, admitted
            if name in self.residents:
                port = self.residents[name]
                self.resident_inflight += 1
            else:
                port = await self.admit(name)
            self.waiting -= 1  # now in flight
            admitted = True
            headers = {k: v for k, v in req.headers.items() if k.lower() not in HOP_HEADERS}
            assert self.session, "app() not started"
            return await self.session.request(
                req.method, f"http://127.0.0.1:{port}{req.rel_url}", data=body, headers=headers
            )

        try:
            body = await req.read()
            try:
                name = self.requested(req, body)
            except SwapFailed as e:
                status = 400
                error = api_error(str(e), type="invalid_request_error", code="model_not_loaded")
                return web.json_response(error, status=status)
            resp, sse, captured = await self.forward(req, body, connect)
            status = resp.status
            if status >= 500:
                self.errors += 1
            if req.method == "POST" and (c := parse_usage(bytes(captured), sse)):
                self.usage.setdefault(c.model or self.presets[self.loaded].alias, Usage()).add(c)
                detail = " " + c.describe()
            return resp
        except Exception:
            self.errors += 1
            log.exception("%s %s -> :%d failed", req.method, req.path, port)
            raise
        finally:
            if admitted and name in self.residents:
                self.resident_inflight -= 1
            elif admitted:
                self.inflight[port] -= 1
            else:
                self.waiting -= 1
            self.last_activity = time.time()
            took = time.time() - start
            log.info("%s %s -> :%d %s %.2fs%s", req.method, req.path, port, status or "-", took, detail)

    def full(self) -> bool:
        """True when max_queue is set and that many requests already wait beyond the slots."""
        limit = self.settings.max_queue
        slots = self.runtime.slots or len(self.inflight)
        return limit is not None and sum(self.inflight.values()) + self.waiting >= slots + limit

    # ----------------------------------------------------------------------- presets and swaps

    def resolve(self, name: object) -> str | None:
        """The preset called `name` (its preset name or alias), or None."""
        if not isinstance(name, str):
            return None
        if name in self.presets:
            return name
        return next((n for n, m in self.presets.items() if m.alias == name), None)

    def requested(self, req: web.Request, body: bytes) -> str | None:
        """The preset a request names in `model` (the JSON body, or ?model= for GET). None for
        no name or an unknown one: those go to the loaded preset, so clients that send any
        name keep working. Raises SwapFailed for another preset when autoload is off."""
        name: object = req.query.get("model")
        if name is None and b'"model"' in body:
            with contextlib.suppress(ValueError):
                doc = json.loads(body)
                name = doc.get("model") if isinstance(doc, dict) else None
        preset = self.resolve(name)
        swappable = preset not in self.residents
        if (
            preset
            and swappable
            and not self.settings.autoload
            and preset not in (self.loaded, self.swap and self.swap.target)
        ):
            loaded = self.presets[self.loaded].alias
            raise SwapFailed(f"model {name} is not loaded ({loaded} is) and autoload is off: `kis use {preset}`")
        return preset

    async def admit(self, name: str | None) -> int:
        """Wait until preset `name` (None: whichever is loaded) serves, then count the request
        in flight on its least busy instance; return that instance's port."""
        await self.ensure(name)
        if not self.inflight:
            raise SwapFailed(self.failure or "no model loaded")
        port = min(self.inflight, key=lambda p: self.inflight[p])
        self.inflight[port] += 1
        return port

    async def ensure(self, name: str | None) -> None:
        """Return once preset `name` (None: whichever is loaded) serves, starting a swap if
        needed; raise SwapFailed if loading it fails. While a swap drains and loads, requests
        for every preset wait, so a busy preset can't keep another one from loading."""
        while True:
            swap = self.swap
            if swap and not (swap.downloading and name in (None, self.loaded)):
                assert swap.task
                error = await asyncio.shield(swap.task)
                if error and swap.target == name:
                    raise SwapFailed(error)
            elif name is None or name == self.loaded:
                return
            else:
                self.swap = Swap(name)
                self.swap.task = asyncio.create_task(self._swap(self.swap))

    async def _swap(self, swap: Swap) -> str | None:
        """Load swap.target: download it while the loaded preset still serves, wait for the
        requests in flight, then replace the instances. Return an error, or None."""
        previous, model, started = self.loaded, self.presets[swap.target], time.time()
        self.notify(
            EventType.MODEL_LOADING, model=model.alias, preset=swap.target, previous=self.presets[previous].alias
        )
        stopped = False
        try:
            await asyncio.to_thread(self.backends.fetch, swap.target)
            swap.downloading = False
            while any(self.inflight.values()):
                await asyncio.sleep(DRAIN_POLL_S)
            stopped = True
            self.inflight = {port: 0 for port in await asyncio.to_thread(self.backends.load, swap.target)}
            self.loaded = swap.target
        except Exception as e:
            log.exception("loading %s failed", swap.target)
            failed: dict[str, Any] = {"model": model.alias, "preset": swap.target, "error": repr(e)[:2000]}
            if isinstance(e, BackendsFailed):
                failed["log"] = e.log
            self.notify(EventType.MODEL_FAILED, **failed)
            if stopped:
                await self.restore(previous)
            return f"loading {model.alias} failed: {e}"
        else:
            seconds = round(time.time() - started)
            self.notify(
                EventType.MODEL_READY,
                **{**dump(self.runtime), "model": model.alias, "preset": swap.target},
                seconds=seconds,
            )
            return None
        finally:
            self.swap = None

    async def restore(self, name: str) -> None:
        """Load preset `name` again after a failed swap; if that fails too, stop the session."""
        try:
            self.inflight = {port: 0 for port in await asyncio.to_thread(self.backends.load, name)}
        except Exception:
            log.exception("reloading %s failed", name)
            self.inflight = {}
            self.failure = f"reloading {self.presets[name].alias} after a failed swap failed"
            self.stop.set()

    def load(self, model: str) -> web.Response:
        """POST /admin/load: start swapping to `model` (a preset name or alias)."""
        name = self.resolve(model)
        if name is None:
            return web.json_response(
                api_error(f"no preset {model!r}; available: {', '.join(self.presets)}"), status=404
            )
        self.last_activity = time.time()
        answer = {"preset": name, "model": self.presets[name].alias, "session": self.settings.session}
        if name in self.residents or (name == self.loaded and not self.swap):
            return web.json_response({**answer, "status": ModelStatus.LOADED.value})

        async def load() -> None:
            with contextlib.suppress(SwapFailed):  # the model_failed event reports it
                await self.ensure(name)

        task = asyncio.create_task(load())
        self.loads.add(task)
        task.add_done_callback(self.loads.discard)
        return web.json_response({**answer, "status": ModelStatus.LOADING.value}, status=202)

    def status(self, name: str) -> ModelStatus:
        if name in self.residents:
            return ModelStatus.LOADED
        if self.swap and self.swap.target == name:
            return ModelStatus.LOADING
        if name == self.loaded and not self.loading:
            return ModelStatus.LOADED
        return ModelStatus.DOWNLOADED if self.backends.downloaded(name) else ModelStatus.AVAILABLE

    def layout(self, name: str) -> dict[str, Any]:
        """Context and slots of a preset in /v1/models: as running for the loaded one, else as
        it would start (slots and, with no `topology` in the preset, topology are only known
        once it runs). `context_length` (OpenRouter's name) is the context each request gets."""
        if name == self.runtime.preset and self.status(name) == ModelStatus.LOADED:
            return {
                "context_length": self.runtime.ctx,
                "parallel": self.runtime.parallel,
                "slots": self.runtime.slots,
                "topology": self.runtime.topology,
            }
        model = self.presets[name]
        if name in self.residents:
            return {
                "context_length": model.ctx or self.settings.ctx,
                "parallel": model.parallel,
                "slots": model.parallel,
            }
        return {
            "context_length": model.ctx or self.settings.ctx,
            "parallel": FITTED.get(name, model.parallel),
            "slots": None,
            "topology": model.topology,
        }

    def models(self) -> dict[str, Any]:
        """/v1/models: every preset of the session, by alias (the name responses report).
        Beyond OpenAI's id/object/created/owned_by, the fields are kis' own (clients ignore them)."""
        data = [
            {
                "id": model.alias,
                "object": "model",
                "created": int(STARTED),
                "owned_by": "kis",
                "preset": name,
                "status": self.status(name).value,
                "resident": name in self.residents,
                **self.layout(name),
            }
            for name, model in self.presets.items()
        ]
        return {"object": "list", "data": data}

    # ----------------------------------------------------------------------- admin

    async def stats(self) -> Stats:
        return Stats(
            session=self.settings.session,
            preset=self.loaded,
            model=self.presets[self.loaded].alias,
            topology=self.runtime.topology,
            slots=self.runtime.slots,
            parallel=self.runtime.parallel,
            ctx=self.runtime.ctx,
            resident=[self.presets[name].alias for name in self.residents],
            uptime_s=round(time.time() - STARTED),
            idle_s=round(self.idle_seconds),
            requests=self.requests,
            inflight=sum(self.inflight.values()),
            errors=self.errors,
            usage={model: u.summary() for model, u in self.usage.items()},
            gpus=await asyncio.to_thread(gpu_stats),
            cpu=await asyncio.to_thread(cpu_stats, self.backends.pids),
            ram=await asyncio.to_thread(ram_stats, self.backends.pids),
        )

    def logs(self, req: web.Request) -> web.Response:
        """Last `lines` lines of server.log or a llama-server log (llama-8090.log, ...)."""
        name = req.query.get("file", SERVER_LOG)
        if not re.fullmatch(r"[\w-]+\.log", name) or not (WORK / name).is_file():
            files = sorted(p.name for p in WORK.glob("*.log"))
            return web.json_response(api_error(f"no log {name!r}; available: {files}"), status=404)
        lines = (WORK / name).read_text(errors="replace").splitlines()[-int(req.query.get("lines", 200)) :]
        return web.Response(text="\n".join(lines) + "\n")

    async def forward(
        self, req: web.Request, body: bytes, connect: Callable[[], Awaitable[aiohttp.ClientResponse]]
    ) -> tuple[web.StreamResponse, bool, bytearray]:
        """Proxy one request: `connect` waits for a swap if needed and sends the request to an
        instance. Stream the response (SSE included) chunk by chunk; return it, whether it is
        SSE, and the tail of its body (for usage stats).

        Cloudflare drops requests whose response doesn't start within 100 s (HTTP 524),
        e.g. a non-streaming completion of a long prompt, or one waiting for a swap. If the
        backend hasn't answered after KEEPALIVE_SECONDS, start a 200 response and send bytes
        clients ignore (JSON whitespace, SSE comments) until it does.
        """
        sse = b'"stream":true' in body.replace(b" ", b"")
        pending = asyncio.ensure_future(connect())
        resp: web.StreamResponse | None = None
        captured = bytearray()
        try:
            while not (await asyncio.wait({pending}, timeout=KEEPALIVE_SECONDS))[0]:
                if resp is None:
                    content_type = "text/event-stream" if sse else "application/json"
                    resp = web.StreamResponse(headers={"Content-Type": content_type})
                    await resp.prepare(req)
                await resp.write(b": keepalive\n\n" if sse else b" ")
            try:
                upstream = pending.result()
            except SwapFailed as e:
                return await self.unavailable(req, resp, sse, str(e)), sse, captured
            if resp is None:
                resp = web.StreamResponse(
                    status=upstream.status,
                    headers={k: v for k, v in upstream.headers.items() if k.lower() not in HOP_HEADERS},
                )
                await resp.prepare(req)
            async for chunk in upstream.content.iter_any():
                await resp.write(chunk)
                captured += chunk
                if len(captured) > CAPTURE_HEAD + CAPTURE_TAIL:
                    del captured[CAPTURE_HEAD:-CAPTURE_TAIL]
            await resp.write_eof()
        except ConnectionResetError:  # client went away; closing upstream stops generation
            pass
        finally:
            if pending.done() and not pending.cancelled() and not pending.exception():
                await pending.result().__aexit__(None, None, None)
            else:
                pending.cancel()
        if resp is None:
            raise ConnectionResetError("upstream closed before responding")
        return resp, sse, captured

    @staticmethod
    async def unavailable(
        req: web.Request, resp: web.StreamResponse | None, sse: bool, message: str
    ) -> web.StreamResponse:
        """A 503 error, or with keepalive bytes already sent, the error as the body of that 200."""
        error = json.dumps(api_error(message, type="unavailable")).encode()
        if resp is None:
            return web.Response(body=error, status=503, content_type="application/json")
        await resp.write(b"data: " + error + b"\n\n" if sse else error)
        await resp.write_eof()
        return resp
