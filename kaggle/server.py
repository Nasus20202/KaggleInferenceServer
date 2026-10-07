"""Kaggle script: serve GGUF models with llama-server behind an
OpenAI-compatible, API-key protected balancer, exposed through a tunnel.

  replicas  model fits one T4 -> one llama-server per GPU (about 2x aggregate tok/s)
  split     larger model      -> one llama-server across both GPUs

One preset is loaded at a time. A request whose `model` names another preset of the
session (autoload), or POST /admin/load, swaps to it: the new preset is downloaded
while the loaded one keeps serving, then requests in flight finish, new ones wait,
and the instances are replaced. Requests without a known model name go to the loaded
preset. GET /v1/models lists every preset with its status.

Started by `kis up <model>`, which fills in CONFIG. Status events, the endpoint
URL and a periodic stats heartbeat go to a private ntfy topic; everything is
also logged with timestamps to the kernel log and /kaggle/working/server.log
(one line per request). Authenticated routes: GET /admin/stats (GPU memory and
load, slots, requests), GET /admin/logs?file=...&lines=N, POST /admin/shutdown.
The session stops after `idle_minutes` without requests, after `max_hours`, or
on /admin/shutdown.
"""

import asyncio
import contextlib
import dataclasses
import enum
import functools
import json
import logging
import os
import re
import subprocess
import sys
import threading
import time
import urllib.request
from collections.abc import AsyncIterator, Awaitable, Callable
from dataclasses import dataclass
from pathlib import Path
from typing import IO, Any, Protocol

import aiohttp
from aiohttp import web

from common import fetch, gpu_stats, llama_command, llama_server_binary, publish  # inlined by `kis`
from kis.schema import (  # inlined by `kis`
    SERVER_LOG,
    Event,
    EventType,
    Model,
    ModelStatus,
    Route,
    ServerConfig,
    Stats,
    Topology,
    Tunnel,
    TunnelKind,
    UsageSummary,
    api_error,
    dump,
    load,
)

# <params> (replaced by `kis up`; to run by hand in the Kaggle UI, edit the rendered .kis/kis-server/server.py)
CONFIG = {
    "api_key": "change-me",
    "model": {
        "repo": "unsloth/Qwen3.5-4B-MTP-GGUF",
        "revision": "86835bf9949e4d14d6860f7910b1340ad4f271a9",
        "file": "Qwen3.5-4B-Q4_K_M.gguf",
        "alias": "Qwen3.5-4B-Q4_K_M",
        "parallel": 16,
    },
    "preset": "qwen35-4b",
    "models": {},
    "autoload": True,
    "prefetch": [],
    "ntfy": {"server": "https://ntfy.sh", "token": ""},
    "ntfy_topic": "",
    "session": "",
    "notify": {"topic": "", "token": "", "server": ""},
    "tunnel": {"kind": "cloudflared"},
    "idle_minutes": 10,
    "max_hours": 11.5,
    "replica_max_gb": 11.0,
    "ctx": 32768,
    "max_queue": None,
    "args": ["--n-gpu-layers", "999", "--flash-attn", "auto", "--metrics"],
}
# </params>

SETTINGS = load(ServerConfig, CONFIG)  # checked and typed; see kis/schema.py for each field
MODEL = SETTINGS.model  # loaded first
PRESETS = SETTINGS.models or {SETTINGS.preset or MODEL.alias: MODEL}  # name -> preset the session can load
START = SETTINGS.preset if SETTINGS.preset in PRESETS else next(iter(PRESETS))
WORK = Path("/kaggle/working")
PORT = 8080  # balancer; the only port the tunnel exposes
BACKEND_PORT = 8090  # llama-server instances use BACKEND_PORT, BACKEND_PORT + 1, ...
KEEPALIVE_SECONDS = 30  # below Cloudflare's 100 s limit for the first response byte
STATS_MINUTES = 10  # heartbeat with GPU and request stats on the ntfy topic
DRAIN_POLL_S = 0.1  # how often a swap checks whether the requests in flight have finished
STARTED = time.time()
log = logging.getLogger("kis")


@dataclass
class Runtime:
    """The loaded preset and the layout of its backends."""

    preset: str = ""
    model: str = ""  # its alias
    topology: Topology = Topology.REPLICAS
    slots: int = 0  # parallel x instances
    parallel: int = 0  # slots per instance
    ctx: int = 0  # tokens per slot


RUNTIME = Runtime()


def setup_logging() -> None:
    formatter = logging.Formatter("%(asctime)s %(levelname)s %(message)s")
    for handler in (logging.StreamHandler(sys.stdout), logging.FileHandler(WORK / SERVER_LOG)):
        handler.setFormatter(formatter)
        log.addHandler(handler)
    log.setLevel(logging.INFO)


class Priority(enum.StrEnum):
    """ntfy message priorities used here."""

    LOW = "low"
    DEFAULT = "default"
    HIGH = "high"


@dataclass(frozen=True)
class Notification:
    """A human-readable message for the phone app (ntfy tags are emoji names)."""

    title: str
    message: str
    priority: Priority = Priority.DEFAULT
    tags: str = ""

    def headers(self) -> dict[str, str]:
        return {"Title": self.title, "Priority": self.priority.value, "Tags": self.tags}


def notify(kind: EventType, **data: object) -> None:
    """Log a status event and publish it to the ntfy topic that `kis` follows."""
    event = Event(kind, t=round(time.time() - STARTED), session=SETTINGS.session, data=data)
    log.info("event %s", json.dumps(event.to_json()))
    try:
        publish(SETTINGS.ntfy, SETTINGS.ntfy_topic, event)
    except OSError as e:
        log.warning("ntfy failed: %s", e)
    if (note := notification(event)) and SETTINGS.notify.topic:
        try:
            publish(SETTINGS.notify.via(SETTINGS.ntfy), SETTINGS.notify.topic, note.message, note.headers())
        except OSError as e:
            log.warning("notification failed: %s", e)


def notification(event: Event) -> Notification | None:
    """The phone notification for events worth one. Never includes the endpoint or keys:
    the notification topic may be public."""
    model = event.data.get("model") or RUNTIME.model or MODEL.alias
    if event.type == EventType.READY:
        layout = f"{RUNTIME.slots} slots x {RUNTIME.ctx} tokens"
        message = f"{layout}, ready after {event.t} s"
        return Notification(f"{model} ready", message, tags="white_check_mark")
    if event.type == EventType.STOPPED:
        return Notification(f"{model} stopped", f"{event.problem} after {round(event.t / 60)} min", tags="stop_sign")
    if event.type == EventType.ERROR:
        return Notification(f"{model} failed", event.problem[:300], Priority.HIGH, "warning")
    if event.type == EventType.RETRY:
        slots = event.data.get("parallel")
        return Notification(f"{model}: out of memory", f"retrying with {slots} slots", Priority.LOW, "hourglass")
    if event.type == EventType.MODEL_READY:
        layout = f"{event.data.get('slots')} slots x {event.data.get('ctx')} tokens"
        return Notification(
            f"{model} loaded", f"{layout}, after {event.data.get('seconds')} s", tags="arrows_counterclockwise"
        )
    if event.type == EventType.MODEL_FAILED:
        return Notification(f"{model} failed to load", event.problem[:300], Priority.HIGH, "warning")
    return None


def log_file(name: str) -> IO[str]:
    """Log file for a child process; it stays open for the process lifetime."""
    return open(WORK / name, "w")


def download(url: str, path: str) -> str:
    if not os.path.exists(path):
        urllib.request.urlretrieve(url, path)
        os.chmod(path, 0o755)
    return path


# --------------------------------------------------------------------------- llama-server instances


FETCHED: dict[str, str] = {}  # repo@revision/file -> local path, downloaded and checked this session
FETCH_LOCKS: dict[str, threading.Lock] = {}  # one download of a file at a time (prefetch and swaps)


def _file_key(model: Model, file: str) -> str:
    return f"{model.repo}@{model.revision}/{file}"


def _fetch_once(model: Model, file: str, sha256: str | None) -> str:
    key = _file_key(model, file)
    with FETCH_LOCKS.setdefault(key, threading.Lock()):
        if key not in FETCHED:
            FETCHED[key] = fetch(model, file, sha256)
        return FETCHED[key]


def model_files(model: Model) -> tuple[str, str | None]:
    """Local paths of a preset's model and draft. Each file is downloaded and its sha256
    checked once per session: hashing the 27B again would add a minute to every swap."""
    draft = _fetch_once(model, model.draft_file, model.draft_sha256) if model.draft_file else None
    return _fetch_once(model, model.file, model.sha256), draft


def downloaded(model: Model) -> bool:
    files = [model.file] + ([model.draft_file] if model.draft_file else [])
    return all(_file_key(model, f) in FETCHED for f in files)


def prefetch(names: list[str]) -> None:
    """Download presets in the background, so swapping to them only has to load them."""
    for name in names:
        try:
            model_files(PRESETS[name])
            log.info("prefetched %s", name)
        except Exception:
            log.exception("prefetch of %s failed", name)


server_binary = functools.cache(llama_server_binary)


def launch(
    binary: str,
    model: Model,
    paths: tuple[str, str | None],
    port: int,
    gpu_ids: list[str],
    parallel: int,
    ctx: int,
) -> subprocess.Popen[bytes]:
    """One llama-server with `parallel` slots of `ctx` tokens each (KV pool = parallel x ctx)."""
    cmd, env = llama_command(binary, model, paths, port, gpu_ids, parallel, ctx, SETTINGS.args)
    return subprocess.Popen(cmd, env=env, stdout=log_file(f"llama-{port}.log"), stderr=subprocess.STDOUT)


def wait_healthy(port: int, proc: subprocess.Popen[bytes], timeout: float = 900) -> bool:
    deadline = time.time() + timeout
    while time.time() < deadline and proc.poll() is None:
        try:
            with urllib.request.urlopen(f"http://127.0.0.1:{port}/health", timeout=5):
                return True
        except OSError:  # connection refused, or 503 while the model loads
            time.sleep(3)
    return False


def plan(
    gpus: list[str], size_gb: float, topology: Topology | None, replica_max_gb: float
) -> tuple[Topology, list[list[str]]]:
    """Choose replicas (one instance per GPU) or split (one instance on all GPUs); return GPU groups."""
    chosen = topology or (Topology.REPLICAS if size_gb <= replica_max_gb else Topology.SPLIT)
    return chosen, [[g] for g in gpus] if chosen == Topology.REPLICAS else [gpus]


OOM = re.compile(r"out of memory|failed to allocate", re.IGNORECASE)
FITTED: dict[str, int] = {}  # preset -> slots per instance that fit, so loading it again skips the retries


class BackendsFailed(RuntimeError):
    """llama-server didn't start; `log` is the tail of its output."""

    def __init__(self, message: str, log: str) -> None:
        super().__init__(message)
        self.log = log


def start_backends(name: str) -> dict[int, subprocess.Popen[bytes]]:
    """Start one llama-server per GPU group for preset `name`. The context per slot is
    fixed; if the KV pool doesn't fit, retry with fewer slots."""
    model = PRESETS[name]
    paths = model_files(model)
    binary = server_binary()
    gpus = subprocess.check_output(["nvidia-smi", "--query-gpu=index", "--format=csv,noheader"], text=True).split()
    size_gb = sum(os.path.getsize(p) for p in paths if p) / 1e9
    topology, groups = plan(gpus, size_gb, model.topology, SETTINGS.replica_max_gb)
    parallel, ctx = FITTED.get(name, model.parallel), model.ctx or SETTINGS.ctx

    ports = range(BACKEND_PORT, BACKEND_PORT + len(groups))
    while True:
        backends = {
            port: launch(binary, model, paths, port, g, parallel, ctx) for port, g in zip(ports, groups, strict=True)
        }
        failed = next((port for port, proc in backends.items() if not wait_healthy(port, proc)), None)
        if failed is None:
            break
        for proc in backends.values():
            proc.terminate()
            proc.wait()
        tail = (WORK / f"llama-{failed}.log").read_text()[-3000:]
        if OOM.search(tail) and parallel > 1:
            parallel -= max(1, parallel // 4)
            notify(EventType.RETRY, reason="out of memory", model=model.alias, parallel=parallel, ctx=ctx)
            continue
        raise BackendsFailed(f"llama-server failed to start with {name} (lower --ctx or --parallel?)", tail)
    FITTED[name] = parallel
    RUNTIME.preset, RUNTIME.model, RUNTIME.topology = name, model.alias, topology
    RUNTIME.slots, RUNTIME.parallel, RUNTIME.ctx = parallel * len(groups), parallel, ctx
    notify(EventType.BACKENDS_READY, model_gb=round(size_gb, 2), **dump(RUNTIME), gpus=dump(gpu_stats()))
    return backends


class BackendSet(Protocol):
    """What the balancer needs of the backends (Backends, or a stand-in in the tests)."""

    @property
    def ports(self) -> list[int]: ...
    def fetch(self, name: str) -> None: ...
    def downloaded(self, name: str) -> bool: ...
    def load(self, name: str) -> list[int]: ...


class Backends:
    """The llama-server instances of the loaded preset. Blocking: the balancer calls
    fetch and load in a thread."""

    def __init__(self) -> None:
        self.procs: dict[int, subprocess.Popen[bytes]] = {}

    @property
    def ports(self) -> list[int]:
        return list(self.procs)

    def fetch(self, name: str) -> None:
        model_files(PRESETS[name])

    def downloaded(self, name: str) -> bool:
        return downloaded(PRESETS[name])

    def load(self, name: str) -> list[int]:
        """Replace the running instances with preset `name`'s; return their ports."""
        self.stop()
        self.procs = start_backends(name)
        return self.ports

    def stop(self) -> None:
        for proc in self.procs.values():
            proc.terminate()
        for proc in self.procs.values():
            try:
                proc.wait(timeout=30)
            except subprocess.TimeoutExpired:
                proc.kill()
                proc.wait()
        self.procs = {}

    def exited(self) -> bool:
        return any(proc.poll() is not None for proc in self.procs.values())


# --------------------------------------------------------------------------- balancer

HOP_HEADERS = {
    "host",
    "authorization",
    "content-length",
    "transfer-encoding",
    "connection",
    "accept-encoding",
    "content-encoding",
}


CAPTURE_BYTES = 256 << 10  # tail of each response kept to read its usage and timings


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
        with contextlib.suppress(ValueError):
            docs.append(json.loads(body))
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
    request to the instance of the loaded preset with the fewest in flight."""

    def __init__(
        self, backends: BackendSet, api_key: str, presets: dict[str, Model] | None = None, loaded: str = ""
    ) -> None:
        self.backends = backends
        self.presets = presets or PRESETS
        self.loaded = loaded or START  # preset name
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
        busy = self.swap or self.waiting or any(self.inflight.values())
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
            if req.query.get("session", SETTINGS.session) != SETTINGS.session:
                return web.json_response({**api_error("other session"), "session": SETTINGS.session}, status=409)
            log.info("shutdown requested")
            self.stop.set()
            return web.json_response({"status": "stopping", "session": SETTINGS.session})
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

        async def connect() -> aiohttp.ClientResponse:
            nonlocal port, admitted
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
            if admitted:
                self.inflight[port] -= 1
            else:
                self.waiting -= 1
            self.last_activity = time.time()
            took = time.time() - start
            log.info("%s %s -> :%d %s %.2fs%s", req.method, req.path, port, status or "-", took, detail)

    def full(self) -> bool:
        """True when max_queue is set and that many requests already wait beyond the slots."""
        limit = SETTINGS.max_queue
        slots = RUNTIME.slots or len(self.inflight)
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
        if preset and not SETTINGS.autoload and preset not in (self.loaded, self.swap and self.swap.target):
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
        notify(EventType.MODEL_LOADING, model=model.alias, preset=swap.target, previous=self.presets[previous].alias)
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
            notify(EventType.MODEL_FAILED, **failed)
            if stopped:
                await self.restore(previous)
            return f"loading {model.alias} failed: {e}"
        else:
            seconds = round(time.time() - started)
            notify(
                EventType.MODEL_READY, **{**dump(RUNTIME), "model": model.alias, "preset": swap.target}, seconds=seconds
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
        answer = {"preset": name, "model": self.presets[name].alias, "session": SETTINGS.session}
        if name == self.loaded and not self.swap:
            return web.json_response({**answer, "status": ModelStatus.LOADED.value})

        async def load() -> None:
            with contextlib.suppress(SwapFailed):  # the model_failed event reports it
                await self.ensure(name)

        task = asyncio.create_task(load())
        self.loads.add(task)
        task.add_done_callback(self.loads.discard)
        return web.json_response({**answer, "status": ModelStatus.LOADING.value}, status=202)

    def status(self, name: str) -> ModelStatus:
        if self.swap and self.swap.target == name:
            return ModelStatus.LOADING
        if name == self.loaded and not self.loading:
            return ModelStatus.LOADED
        return ModelStatus.DOWNLOADED if self.backends.downloaded(name) else ModelStatus.AVAILABLE

    def layout(self, name: str) -> dict[str, Any]:
        """Context and slots of a preset in /v1/models: as running for the loaded one, else as
        it would start (slots and, with no `topology` in the preset, topology are only known
        once it runs). `context_length` (OpenRouter's name) is the context each request gets."""
        if name == RUNTIME.preset and self.status(name) == ModelStatus.LOADED:
            return {
                "context_length": RUNTIME.ctx,
                "parallel": RUNTIME.parallel,
                "slots": RUNTIME.slots,
                "topology": RUNTIME.topology,
            }
        model = self.presets[name]
        return {
            "context_length": model.ctx or SETTINGS.ctx,
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
                **self.layout(name),
            }
            for name, model in self.presets.items()
        ]
        return {"object": "list", "data": data}

    # ----------------------------------------------------------------------- admin

    async def stats(self) -> Stats:
        return Stats(
            session=SETTINGS.session,
            model=self.presets[self.loaded].alias,
            preset=self.loaded,
            topology=RUNTIME.topology,
            slots=RUNTIME.slots,
            parallel=RUNTIME.parallel,
            ctx=RUNTIME.ctx,
            uptime_s=round(time.time() - STARTED),
            idle_s=round(self.idle_seconds),
            inflight=sum(self.inflight.values()),
            requests=self.requests,
            errors=self.errors,
            usage={model: u.summary() for model, u in self.usage.items()},
            gpus=await asyncio.to_thread(gpu_stats),
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
                del captured[:-CAPTURE_BYTES]
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


# --------------------------------------------------------------------------- tunnels


def cloudflared(tunnel: Tunnel, on_ready: Callable[[str], None]) -> None:
    """Quick tunnel (random trycloudflare.com URL) or, with a token, a named tunnel.

    A named tunnel's public hostname must point to http://localhost:8080.
    Runs forever, restarting the tunnel if it drops.
    """
    binary = download(
        "https://github.com/cloudflare/cloudflared/releases/latest/download/cloudflared-linux-amd64", "/tmp/cloudflared"
    )
    cmd = [binary, "tunnel", "--no-autoupdate", "--protocol", "http2"]
    cmd += ["run", "--token", tunnel.token] if tunnel.token else ["--url", f"http://127.0.0.1:{PORT}"]
    while True:
        proc = subprocess.Popen(cmd, stdout=subprocess.PIPE, stderr=subprocess.STDOUT, text=True)
        announced = False
        for line in proc.stdout or []:
            log.info("cloudflared: %s", line.rstrip())
            quick = re.search(r"https://[a-z0-9-]+\.trycloudflare\.com", line)
            url = quick.group(0) if quick else (tunnel.public_url if "Registered tunnel connection" in line else None)
            if url and not announced:
                announced = True
                on_ready(url)
        notify(EventType.TUNNEL_RESTART, code=proc.wait())
        time.sleep(5)


def tailscale(tunnel: Tunnel, on_ready: Callable[[str], None]) -> None:
    """Join the tailnet in userspace mode; inbound tailnet connections reach localhost:<port>."""
    root = "/tmp/tailscale"
    if not os.path.exists(f"{root}/tailscale"):
        os.makedirs(root, exist_ok=True)
        subprocess.run(
            f"curl -fsSL https://pkgs.tailscale.com/stable/tailscale_latest_amd64.tgz"
            f" | tar xz -C {root} --strip-components=1",
            shell=True,
            check=True,
        )
    ts = [f"{root}/tailscale", "--socket=/tmp/tailscaled.sock"]
    subprocess.Popen(
        [f"{root}/tailscaled", "--tun=userspace-networking", "--state=mem:", "--socket=/tmp/tailscaled.sock"],
        stdout=log_file("tailscaled.log"),
        stderr=subprocess.STDOUT,
    )
    time.sleep(3)
    subprocess.run(
        [
            *ts,
            "up",
            f"--authkey={tunnel.authkey}",
            f"--hostname={tunnel.hostname}",
            "--accept-dns=false",
        ],
        check=True,
        timeout=120,
    )
    me = json.loads(subprocess.check_output([*ts, "status", "--json"]))["Self"]
    on_ready(f"http://{me['DNSName'].rstrip('.') or me['TailscaleIPs'][0]}:{PORT}")


TUNNELS = {TunnelKind.CLOUDFLARED: cloudflared, TunnelKind.TAILSCALE: tailscale}


# --------------------------------------------------------------------------- main


async def serve(backends: Backends) -> str:
    """Run the balancer and tunnel until shutdown; return the stop reason."""
    balancer = Balancer(backends, SETTINGS.api_key)
    await balancer.start()
    tunnel = SETTINGS.tunnel

    def on_ready(url: str) -> None:
        notify(EventType.READY, endpoint=url, model=RUNTIME.model, preset=RUNTIME.preset, tunnel=tunnel.kind)

    threading.Thread(target=TUNNELS[tunnel.kind], args=(tunnel, on_ready), daemon=True).start()
    if SETTINGS.prefetch:
        threading.Thread(target=prefetch, args=(SETTINGS.prefetch,), daemon=True).start()

    reason, last_stats = "shutdown requested", time.time()
    while not balancer.stop.is_set():
        await asyncio.sleep(20)
        if time.time() - last_stats > STATS_MINUTES * 60:
            last_stats = time.time()
            notify(EventType.STATS, **dump(await balancer.stats()))
        if balancer.idle_seconds > SETTINGS.idle_minutes * 60:
            reason = f"idle for {SETTINGS.idle_minutes:g} min"
        elif time.time() - STARTED > SETTINGS.max_hours * 3600:
            reason = "max_hours reached"
        elif not balancer.loading and backends.exited():
            reason = "llama-server exited"
        else:
            continue
        break
    await balancer.close()
    return balancer.failure or reason


def main() -> None:
    setup_logging()
    notify(EventType.STARTING, model=MODEL.alias, preset=START)
    backends = Backends()
    backends.fetch(START)
    notify(EventType.DOWNLOADED)
    try:
        backends.load(START)
    except BackendsFailed as e:
        notify(EventType.ERROR, error=str(e), log=e.log)
        sys.exit(1)
    try:
        reason = asyncio.run(serve(backends))
    finally:
        backends.stop()
    notify(EventType.STOPPED, reason=reason)


if __name__ == "__main__":
    try:
        main()
    except Exception as e:
        notify(EventType.ERROR, error=repr(e)[:2000])
        raise
