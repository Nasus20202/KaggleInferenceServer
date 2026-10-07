"""Kaggle script: serve one GGUF model with llama-server behind an
OpenAI-compatible, API-key protected balancer, exposed through a tunnel.

  replicas  model fits one T4 -> one llama-server per GPU (about 2x aggregate tok/s)
  split     larger model      -> one llama-server across both GPUs

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
import json
import logging
import os
import re
import subprocess
import sys
import threading
import time
import urllib.request
from collections.abc import AsyncIterator, Callable
from dataclasses import dataclass
from pathlib import Path
from typing import IO

import aiohttp
from aiohttp import web

from common import fetch, gpu_stats, llama_command, llama_server_binary, publish  # inlined by `kis`
from kis.schema import (  # inlined by `kis`
    SERVER_LOG,
    Event,
    EventType,
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
MODEL = SETTINGS.model
WORK = Path("/kaggle/working")
PORT = 8080  # balancer; the only port the tunnel exposes
BACKEND_PORT = 8090  # llama-server instances use BACKEND_PORT, BACKEND_PORT + 1, ...
KEEPALIVE_SECONDS = 30  # below Cloudflare's 100 s limit for the first response byte
STATS_MINUTES = 10  # heartbeat with GPU and request stats on the ntfy topic
STARTED = time.time()
log = logging.getLogger("kis")


@dataclass
class Runtime:
    """Layout of the running backends."""

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
    model = MODEL.alias
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


def launch(
    binary: str, model_path: str, draft_path: str | None, port: int, gpu_ids: list[str], parallel: int, ctx: int
) -> subprocess.Popen[bytes]:
    """One llama-server with `parallel` slots of `ctx` tokens each (KV pool = parallel x ctx)."""
    cmd, env = llama_command(binary, MODEL, (model_path, draft_path), port, gpu_ids, parallel, ctx, SETTINGS.args)
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


def start_backends(model_path: str, draft_path: str | None) -> dict[int, subprocess.Popen[bytes]]:
    """Start one llama-server per GPU group. The context per slot is fixed; if the KV pool
    doesn't fit, retry with fewer slots."""
    binary = llama_server_binary()
    gpus = subprocess.check_output(["nvidia-smi", "--query-gpu=index", "--format=csv,noheader"], text=True).split()
    size_gb = sum(os.path.getsize(p) for p in (model_path, draft_path) if p) / 1e9
    topology, groups = plan(gpus, size_gb, MODEL.topology, SETTINGS.replica_max_gb)
    parallel, ctx = MODEL.parallel, MODEL.ctx or SETTINGS.ctx

    ports = range(BACKEND_PORT, BACKEND_PORT + len(groups))
    while True:
        backends = {
            port: launch(binary, model_path, draft_path, port, g, parallel, ctx)
            for port, g in zip(ports, groups, strict=True)
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
            notify(EventType.RETRY, reason="out of memory", parallel=parallel, ctx=ctx)
            continue
        notify(EventType.ERROR, error="llama-server failed to start (lower --ctx or --parallel?)", log=tail)
        raise RuntimeError("llama-server failed to start")
    RUNTIME.topology, RUNTIME.slots, RUNTIME.parallel, RUNTIME.ctx = topology, parallel * len(groups), parallel, ctx
    notify(EventType.BACKENDS_READY, model_gb=round(size_gb, 2), **dump(RUNTIME), gpus=dump(gpu_stats()))
    return backends


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


class Balancer:
    """Checks the API key and sends each request to the instance with the fewest in flight."""

    def __init__(self, ports: list[int], api_key: str) -> None:
        self.inflight = {port: 0 for port in ports}
        self.api_key = api_key
        self.last_activity = time.time()
        self.stop = asyncio.Event()
        self.requests = self.errors = 0
        self.usage: dict[str, Usage] = {}
        self.session: aiohttp.ClientSession | None = None
        self.runner: web.AppRunner | None = None

    @property
    def idle_seconds(self) -> float:
        return 0 if any(self.inflight.values()) else time.time() - self.last_activity

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
        if self.full():
            log.warning("%s %s 429 (all slots busy, queue full)", req.method, req.path)
            return web.json_response(
                api_error("server busy: all slots in use and the queue is full", type="rate_limit"),
                status=429,
                headers={"Retry-After": "5"},
            )
        port = min(self.inflight, key=lambda p: self.inflight[p])
        self.inflight[port] += 1
        self.requests += 1
        self.last_activity = start = time.time()
        status, detail = 0, ""
        try:
            resp, sse, captured = await self.forward(req, f"http://127.0.0.1:{port}{req.rel_url}")
            status = resp.status
            if status >= 500:
                self.errors += 1
            if req.method == "POST" and (c := parse_usage(bytes(captured), sse)):
                self.usage.setdefault(c.model or MODEL.alias, Usage()).add(c)
                detail = " " + c.describe()
            return resp
        except Exception:
            self.errors += 1
            log.exception("%s %s -> :%d failed", req.method, req.path, port)
            raise
        finally:
            self.inflight[port] -= 1
            self.last_activity = time.time()
            took = time.time() - start
            log.info("%s %s -> :%d %s %.2fs%s", req.method, req.path, port, status or "-", took, detail)

    def full(self) -> bool:
        """True when max_queue is set and that many requests already wait beyond the slots."""
        limit = SETTINGS.max_queue
        slots = RUNTIME.slots or len(self.inflight)
        return limit is not None and sum(self.inflight.values()) >= slots + limit

    async def stats(self) -> Stats:
        return Stats(
            session=SETTINGS.session,
            model=MODEL.alias,
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

    async def forward(self, req: web.Request, url: str) -> tuple[web.StreamResponse, bool, bytearray]:
        """Proxy one request, streaming the response (SSE included) chunk by chunk; return
        the response, whether it is SSE, and the tail of its body (for usage stats).

        Cloudflare drops requests whose response doesn't start within 100 s (HTTP 524),
        e.g. a non-streaming completion of a long prompt. If the backend hasn't answered
        after KEEPALIVE_SECONDS, start a 200 response and send bytes clients ignore
        (JSON whitespace, SSE comments) until it does.
        """
        headers = {k: v for k, v in req.headers.items() if k.lower() not in HOP_HEADERS}
        body = await req.read()
        sse = b'"stream":true' in body.replace(b" ", b"")
        assert self.session, "app() not started"
        request = self.session.request(req.method, url, data=body, headers=headers)
        pending = asyncio.ensure_future(request.__aenter__())
        resp: web.StreamResponse | None = None
        captured = bytearray()
        try:
            while not (await asyncio.wait({pending}, timeout=KEEPALIVE_SECONDS))[0]:
                if resp is None:
                    content_type = "text/event-stream" if sse else "application/json"
                    resp = web.StreamResponse(headers={"Content-Type": content_type})
                    await resp.prepare(req)
                await resp.write(b": keepalive\n\n" if sse else b" ")
            upstream = pending.result()
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
                await request.__aexit__(None, None, None)
            else:
                pending.cancel()
        if resp is None:
            raise ConnectionResetError("upstream closed before responding")
        return resp, sse, captured


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


async def serve(backends: dict[int, subprocess.Popen[bytes]]) -> str:
    """Run the balancer and tunnel until shutdown; return the stop reason."""
    balancer = Balancer(list(backends), SETTINGS.api_key)
    await balancer.start()
    tunnel = SETTINGS.tunnel

    def on_ready(url: str) -> None:
        notify(EventType.READY, endpoint=url, model=MODEL.alias, tunnel=tunnel.kind)

    threading.Thread(target=TUNNELS[tunnel.kind], args=(tunnel, on_ready), daemon=True).start()

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
        elif any(proc.poll() is not None for proc in backends.values()):
            reason = "llama-server exited"
        else:
            continue
        break
    await balancer.close()
    return reason


def main() -> None:
    setup_logging()
    notify(EventType.STARTING, model=MODEL.alias)
    model_path = fetch(MODEL, MODEL.file, MODEL.sha256)
    draft_path = fetch(MODEL, MODEL.draft_file, MODEL.draft_sha256) if MODEL.draft_file else None
    notify(EventType.DOWNLOADED)
    backends = start_backends(model_path, draft_path)
    try:
        reason = asyncio.run(serve(backends))
    finally:
        for proc in backends.values():
            proc.terminate()
    notify(EventType.STOPPED, reason=reason)


if __name__ == "__main__":
    try:
        main()
    except Exception as e:
        notify(EventType.ERROR, error=repr(e)[:2000])
        raise
