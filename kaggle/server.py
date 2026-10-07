"""Kaggle script: serve one GGUF model with llama-server behind an
OpenAI-compatible, API-key protected balancer, exposed through a tunnel.

  replicas  model fits one T4 -> one llama-server per GPU (about 2x aggregate tok/s)
  split     larger model      -> one llama-server across both GPUs

Started by `kis up <model>`, which fills in CONFIG. Status events, the endpoint
URL and a periodic stats heartbeat go to a private ntfy.sh topic; everything is
also logged with timestamps to the kernel log and /kaggle/working/server.log
(one line per request). Authenticated routes: GET /admin/stats (GPU memory and
load, slots, requests), GET /admin/logs?file=...&lines=N, POST /admin/shutdown.
The session stops after `idle_minutes` without requests, after `max_hours`, or
on /admin/shutdown.
"""

import asyncio
import contextlib
import hashlib
import json
import logging
import os
import re
import shutil
import subprocess
import sys
import threading
import time
import urllib.request
from pathlib import Path

import aiohttp
from aiohttp import web

# <params> (replaced by `kis up`; edit by hand to run the script in the Kaggle UI)
CONFIG = {
    "api_key": "change-me",
    "ntfy_topic": "",
    "session": "",  # id in every event; `kis` tells sessions apart during a rollover
    "tunnel": {"kind": "cloudflared"},
    "idle_minutes": 30,
    "max_hours": 11.5,
    "replica_max_gb": 11.0,
    "ctx": 32768,
    "max_queue": None,  # requests waiting beyond the slots before 429; None = no limit
    "args": ["--n-gpu-layers", "999", "--flash-attn", "auto", "--metrics"],
    "model": {
        "repo": "unsloth/Qwen3.5-4B-MTP-GGUF",
        "revision": "86835bf9949e4d14d6860f7910b1340ad4f271a9",
        "file": "Qwen3.5-4B-Q4_K_M.gguf",
        "alias": "Qwen3.5-4B-Q4_K_M",
        "parallel": 16,
    },
}
# </params>

MODEL = CONFIG["model"]
WORK = Path("/kaggle/working")
MODELS_DIR = Path("/kaggle/tmp/models" if os.path.isdir("/kaggle/tmp") else "/tmp/models")
PORT = 8080  # balancer; the only port the tunnel exposes
BACKEND_PORT = 8090  # llama-server instances use BACKEND_PORT, BACKEND_PORT + 1, ...
KEEPALIVE_SECONDS = 30  # below Cloudflare's 100 s limit for the first response byte
STATS_MINUTES = 10  # heartbeat with GPU and request stats on the ntfy topic
STARTED = time.time()
RUNTIME = {}  # topology, slots, parallel and ctx of the running backends
log = logging.getLogger("kis")


def setup_logging():
    formatter = logging.Formatter("%(asctime)s %(levelname)s %(message)s")
    for handler in (logging.StreamHandler(sys.stdout), logging.FileHandler(WORK / "server.log")):
        handler.setFormatter(formatter)
        log.addHandler(handler)
    log.setLevel(logging.INFO)


def notify(event: str, **data):
    """Log a status event and publish it to the ntfy topic that `kis` follows."""
    msg = json.dumps({"event": event, "t": round(time.time() - STARTED), "session": CONFIG["session"], **data})
    log.info("event %s", msg)
    if CONFIG["ntfy_topic"]:
        try:
            urllib.request.urlopen(f"https://ntfy.sh/{CONFIG['ntfy_topic']}", data=msg.encode(), timeout=10)
        except OSError as e:
            log.warning("ntfy failed: %s", e)


def log_file(name: str):
    """Log file for a child process; it stays open for the process lifetime."""
    return open(WORK / name, "w")


def download(url: str, path: str) -> str:
    if not os.path.exists(path):
        urllib.request.urlretrieve(url, path)
        os.chmod(path, 0o755)
    return path


# --------------------------------------------------------------------------- model


def llama_server_binary() -> str:
    """Copy llama-server from the attached build output (inputs are read-only)."""
    found = sorted(Path("/kaggle/input").rglob("llama-server"))
    if not found:
        raise RuntimeError("llama-server not found: attach the build kernel output (`kis build`)")
    shutil.copy(found[0], "/tmp/llama-server")
    os.chmod("/tmp/llama-server", 0o755)
    return "/tmp/llama-server"


def fetch(file: str, sha256: str | None) -> str:
    from huggingface_hub import hf_hub_download

    os.environ.setdefault("HF_XET_HIGH_PERFORMANCE", "1")
    path = hf_hub_download(MODEL["repo"], file, revision=MODEL.get("revision"), local_dir=MODELS_DIR)
    if sha256:
        digest = hashlib.sha256()
        with open(path, "rb") as f:
            for chunk in iter(lambda: f.read(1 << 24), b""):
                digest.update(chunk)
        if digest.hexdigest() != sha256:
            raise RuntimeError(f"sha256 mismatch for {file}")
    return path


# --------------------------------------------------------------------------- llama-server instances


def launch(
    binary: str, model_path: str, draft_path: str | None, port: int, gpu_ids: list[str], parallel: int, ctx: int
) -> subprocess.Popen:
    """One llama-server with `parallel` slots of `ctx` tokens each (KV pool = parallel x ctx)."""
    cmd = [
        binary,
        "-m",
        model_path,
        "--host",
        "127.0.0.1",
        "--port",
        str(port),
        "--alias",
        MODEL["alias"],
        "--parallel",
        str(parallel),
        "--kv-unified-per-slot",
        str(ctx),
    ]
    if draft_path:
        cmd += ["--model-draft", draft_path]
    if len(gpu_ids) > 1:
        cmd += ["--split-mode", "layer", "--tensor-split", MODEL.get("tensor_split") or ",".join("1" * len(gpu_ids))]
    cmd += CONFIG["args"] + MODEL.get("args", [])
    env = {**os.environ, "CUDA_VISIBLE_DEVICES": ",".join(gpu_ids)}
    return subprocess.Popen(cmd, env=env, stdout=log_file(f"llama-{port}.log"), stderr=subprocess.STDOUT)


def wait_healthy(port: int, proc: subprocess.Popen, timeout: float = 900) -> bool:
    deadline = time.time() + timeout
    while time.time() < deadline and proc.poll() is None:
        try:
            with urllib.request.urlopen(f"http://127.0.0.1:{port}/health", timeout=5):
                return True
        except OSError:  # connection refused, or 503 while the model loads
            time.sleep(3)
    return False


def plan(gpus: list[str], size_gb: float, topology: str | None, replica_max_gb: float) -> tuple[str, list[list[str]]]:
    """Choose replicas (one instance per GPU) or split (one instance on all GPUs); return GPU groups."""
    topology = topology or ("replicas" if size_gb <= replica_max_gb else "split")
    return topology, [[g] for g in gpus] if topology == "replicas" else [gpus]


OOM = re.compile(r"out of memory|failed to allocate", re.IGNORECASE)


def gpu_stats() -> list[dict]:
    """Memory (MiB) and utilization (%) of each GPU; empty if nvidia-smi fails."""
    try:
        out = subprocess.check_output(
            [
                "nvidia-smi",
                "--query-gpu=index,memory.used,memory.total,utilization.gpu",
                "--format=csv,noheader,nounits",
            ],
            text=True,
            timeout=15,
        )
    except (OSError, subprocess.SubprocessError) as e:
        log.warning("nvidia-smi failed: %s", e)
        return []
    keys = ("gpu", "used_mib", "total_mib", "util_pct")
    return [dict(zip(keys, map(int, line.split(", ")), strict=True)) for line in out.strip().splitlines()]


def start_backends(model_path: str, draft_path: str | None) -> dict[int, subprocess.Popen]:
    """Start one llama-server per GPU group. The context per slot is fixed; if the KV pool
    doesn't fit, retry with fewer slots."""
    binary = llama_server_binary()
    gpus = subprocess.check_output(["nvidia-smi", "--query-gpu=index", "--format=csv,noheader"], text=True).split()
    size_gb = sum(os.path.getsize(p) for p in (model_path, draft_path) if p) / 1e9
    topology, groups = plan(gpus, size_gb, MODEL.get("topology"), CONFIG["replica_max_gb"])
    parallel, ctx = MODEL["parallel"], MODEL.get("ctx") or CONFIG["ctx"]

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
        log = (WORK / f"llama-{failed}.log").read_text()[-3000:]
        if OOM.search(log) and parallel > 1:
            parallel -= max(1, parallel // 4)
            notify("retry", reason="out of memory", parallel=parallel, ctx=ctx)
            continue
        notify("error", error="llama-server failed to start (lower --ctx or --parallel?)", log=log)
        raise RuntimeError("llama-server failed to start")
    RUNTIME.update(topology=topology, slots=parallel * len(groups), parallel=parallel, ctx=ctx)
    notify("backends_ready", model_gb=round(size_gb, 2), **RUNTIME, gpus=gpu_stats())
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


def parse_usage(body: bytes, sse: bool) -> dict | None:
    """Tokens and timings of one completion from llama-server's `usage` and `timings`
    (in the JSON body, or in the last events of an SSE stream); None if absent."""
    docs = []
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
    usage = next((d["usage"] for d in docs if isinstance(d, dict) and d.get("usage")), None)
    timings = next((d["timings"] for d in docs if isinstance(d, dict) and d.get("timings")), None)
    if not usage and not timings:
        return None
    usage, timings = usage or {}, timings or {}
    cached = (usage.get("prompt_tokens_details") or {}).get("cached_tokens", timings.get("cache_n", 0))
    return {
        "model": next((d["model"] for d in docs if isinstance(d, dict) and d.get("model")), None),
        "in": usage.get("prompt_tokens", timings.get("prompt_n", 0) + timings.get("cache_n", 0)),
        "out": usage.get("completion_tokens", timings.get("predicted_n", 0)),
        "cached": cached or 0,
        "prefill_n": timings.get("prompt_n", 0),
        "prefill_ms": timings.get("prompt_ms", 0.0),
        "decode_ms": timings.get("predicted_ms", 0.0),
        "draft_n": timings.get("draft_n", 0),
        "draft_accepted": timings.get("draft_n_accepted", 0),
    }


def describe(u: dict) -> str:
    """One log fragment: tokens and speeds of a completion."""
    text = f"in={u['in']} cached={u['cached']} out={u['out']}"
    if u["prefill_ms"]:
        text += f" prefill={u['prefill_n'] / u['prefill_ms'] * 1000:.0f}tok/s"
    if u["decode_ms"]:
        text += f" decode={u['out'] / u['decode_ms'] * 1000:.1f}tok/s"
    if u["draft_n"]:
        text += f" draft={u['draft_accepted'] / u['draft_n']:.0%}"
    return text


class Usage:
    """Running totals of completions for one model."""

    FIELDS = ("in", "out", "cached", "prefill_n", "prefill_ms", "decode_ms", "draft_n", "draft_accepted")

    def __init__(self):
        self.requests = 0
        self.totals = dict.fromkeys(self.FIELDS, 0)

    def add(self, u: dict):
        self.requests += 1
        for key in self.FIELDS:
            self.totals[key] += u[key]

    def summary(self) -> dict:
        t = self.totals
        return {
            "requests": self.requests,
            "tokens_in": t["in"],
            "tokens_out": t["out"],
            "tokens_cached": t["cached"],
            "cache_rate": round(t["cached"] / t["in"], 3) if t["in"] else None,
            "prefill_tps": round(t["prefill_n"] / t["prefill_ms"] * 1000, 1) if t["prefill_ms"] else None,
            "decode_tps": round(t["out"] / t["decode_ms"] * 1000, 1) if t["decode_ms"] else None,
            "draft_acceptance": round(t["draft_accepted"] / t["draft_n"], 3) if t["draft_n"] else None,
        }


class Balancer:
    """Checks the API key and sends each request to the instance with the fewest in flight."""

    def __init__(self, ports: list[int], api_key: str):
        self.inflight = {port: 0 for port in ports}
        self.api_key = api_key
        self.last_activity = time.time()
        self.stop = asyncio.Event()
        self.requests = self.errors = 0
        self.usage: dict[str, Usage] = {}

    @property
    def idle_seconds(self) -> float:
        return 0 if any(self.inflight.values()) else time.time() - self.last_activity

    def app(self) -> web.Application:
        async def client_session(app: web.Application):
            self.session = aiohttp.ClientSession(timeout=aiohttp.ClientTimeout(total=None))
            yield
            await self.session.close()

        app = web.Application(client_max_size=256 << 20)
        app.cleanup_ctx.append(client_session)
        app.router.add_route("*", "/{tail:.*}", self.handle)
        return app

    async def start(self):
        self.runner = web.AppRunner(self.app())
        await self.runner.setup()
        await web.TCPSite(self.runner, "127.0.0.1", PORT).start()

    async def close(self):
        await self.runner.cleanup()

    async def handle(self, req: web.Request) -> web.StreamResponse:
        if req.path == "/health":
            return web.json_response({"status": "ok"})
        if req.headers.get("Authorization") != f"Bearer {self.api_key}":
            return web.json_response({"error": {"message": "invalid api key"}}, status=401)
        if req.path == "/admin/shutdown":
            # With a named tunnel, two sessions share one URL during a rollover: only stop the one asked for.
            if req.query.get("session", CONFIG["session"]) != CONFIG["session"]:
                return web.json_response({"error": {"message": "other session"}, "session": CONFIG["session"]}, 409)
            log.info("shutdown requested")
            self.stop.set()
            return web.json_response({"status": "stopping", "session": CONFIG["session"]})
        if req.path == "/admin/stats":
            return web.json_response(await self.stats())
        if req.path == "/admin/logs":
            return self.logs(req)
        if self.full():
            log.warning("%s %s 429 (all slots busy, queue full)", req.method, req.path)
            return web.json_response(
                {"error": {"message": "server busy: all slots in use and the queue is full", "type": "rate_limit"}},
                status=429,
                headers={"Retry-After": "5"},
            )
        port = min(self.inflight, key=lambda p: self.inflight[p])
        self.inflight[port] += 1
        self.requests += 1
        self.last_activity = start = time.time()
        status, detail = "-", ""
        try:
            resp, sse, captured = await self.forward(req, f"http://127.0.0.1:{port}{req.rel_url}")
            status = resp.status
            if status >= 500:
                self.errors += 1
            if req.method == "POST" and (u := parse_usage(bytes(captured), sse)):
                self.usage.setdefault(u["model"] or MODEL["alias"], Usage()).add(u)
                detail = " " + describe(u)
            return resp
        except Exception:
            self.errors += 1
            log.exception("%s %s -> :%d failed", req.method, req.path, port)
            raise
        finally:
            self.inflight[port] -= 1
            self.last_activity = time.time()
            log.info("%s %s -> :%d %s %.2fs%s", req.method, req.path, port, status, time.time() - start, detail)

    def full(self) -> bool:
        """True when max_queue is set and that many requests already wait beyond the slots."""
        limit = CONFIG.get("max_queue")
        slots = RUNTIME.get("slots", len(self.inflight))
        return limit is not None and sum(self.inflight.values()) >= slots + limit

    async def stats(self) -> dict:
        return {
            "session": CONFIG["session"],
            "model": MODEL["alias"],
            **RUNTIME,
            "uptime_s": round(time.time() - STARTED),
            "idle_s": round(self.idle_seconds),
            "inflight": sum(self.inflight.values()),
            "requests": self.requests,
            "errors": self.errors,
            "usage": {model: u.summary() for model, u in self.usage.items()},
            "gpus": await asyncio.to_thread(gpu_stats),
        }

    def logs(self, req: web.Request) -> web.Response:
        """Last `lines` lines of server.log or a llama-server log (llama-8090.log, ...)."""
        name = req.query.get("file", "server.log")
        if not re.fullmatch(r"[\w-]+\.log", name) or not (WORK / name).is_file():
            files = sorted(p.name for p in WORK.glob("*.log"))
            return web.json_response({"error": {"message": f"no log {name!r}; available: {files}"}}, status=404)
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
        request = self.session.request(req.method, url, data=body, headers=headers)
        pending = asyncio.ensure_future(request.__aenter__())
        resp, captured = None, bytearray()
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
        return resp, sse, captured


# --------------------------------------------------------------------------- tunnels


def cloudflared(tunnel: dict, on_ready):
    """Quick tunnel (random trycloudflare.com URL) or, with a token, a named tunnel.

    A named tunnel's public hostname must point to http://localhost:8080.
    Runs forever, restarting the tunnel if it drops.
    """
    binary = download(
        "https://github.com/cloudflare/cloudflared/releases/latest/download/cloudflared-linux-amd64", "/tmp/cloudflared"
    )
    cmd = [binary, "tunnel", "--no-autoupdate", "--protocol", "http2"]
    cmd += ["run", "--token", tunnel["token"]] if tunnel.get("token") else ["--url", f"http://127.0.0.1:{PORT}"]
    while True:
        proc = subprocess.Popen(cmd, stdout=subprocess.PIPE, stderr=subprocess.STDOUT, text=True)
        announced = False
        for line in proc.stdout or []:
            log.info("cloudflared: %s", line.rstrip())
            quick = re.search(r"https://[a-z0-9-]+\.trycloudflare\.com", line)
            url = (
                quick.group(0)
                if quick
                else (tunnel.get("public_url") if "Registered tunnel connection" in line else None)
            )
            if url and not announced:
                announced = True
                on_ready(url)
        notify("tunnel_restart", code=proc.wait())
        time.sleep(5)


def tailscale(tunnel: dict, on_ready):
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
            f"--authkey={tunnel['authkey']}",
            f"--hostname={tunnel.get('hostname', 'kaggle-llm')}",
            "--accept-dns=false",
        ],
        check=True,
        timeout=120,
    )
    me = json.loads(subprocess.check_output([*ts, "status", "--json"]))["Self"]
    on_ready(f"http://{me['DNSName'].rstrip('.') or me['TailscaleIPs'][0]}:{PORT}")


TUNNELS = {"cloudflared": cloudflared, "tailscale": tailscale}


# --------------------------------------------------------------------------- main


async def serve(backends: dict[int, subprocess.Popen]) -> str:
    """Run the balancer and tunnel until shutdown; return the stop reason."""
    balancer = Balancer(list(backends), CONFIG["api_key"])
    await balancer.start()
    tunnel = CONFIG["tunnel"]
    on_ready = lambda url: notify("ready", endpoint=url, model=MODEL["alias"], tunnel=tunnel["kind"])  # noqa: E731
    threading.Thread(target=TUNNELS[tunnel["kind"]], args=(tunnel, on_ready), daemon=True).start()

    reason, last_stats = "shutdown requested", time.time()
    while not balancer.stop.is_set():
        await asyncio.sleep(20)
        if time.time() - last_stats > STATS_MINUTES * 60:
            last_stats = time.time()
            notify("stats", **await balancer.stats())
        if balancer.idle_seconds > CONFIG["idle_minutes"] * 60:
            reason = f"idle for {CONFIG['idle_minutes']} min"
        elif time.time() - STARTED > CONFIG["max_hours"] * 3600:
            reason = "max_hours reached"
        elif any(proc.poll() is not None for proc in backends.values()):
            reason = "llama-server exited"
        else:
            continue
        break
    await balancer.close()
    return reason


def main():
    setup_logging()
    notify("starting", model=MODEL["alias"])
    model_path = fetch(MODEL["file"], MODEL.get("sha256"))
    draft_path = fetch(MODEL["draft_file"], MODEL.get("draft_sha256")) if MODEL.get("draft_file") else None
    notify("downloaded")
    backends = start_backends(model_path, draft_path)
    try:
        reason = asyncio.run(serve(backends))
    finally:
        for proc in backends.values():
            proc.terminate()
    notify("stopped", reason=reason)


if __name__ == "__main__":
    try:
        main()
    except Exception as e:
        notify("error", error=repr(e)[:2000])
        raise
