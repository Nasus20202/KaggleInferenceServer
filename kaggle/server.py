"""Kaggle script: serve one GGUF model with llama-server behind an
OpenAI-compatible, API-key protected balancer, exposed through a tunnel.

  replicas  model fits one T4 -> one llama-server per GPU (about 2x aggregate tok/s)
  split     larger model      -> one llama-server across both GPUs

Started by `kis up <model>`, which fills in CONFIG. Status events and the
endpoint URL go to a private ntfy.sh topic. The session stops after
`idle_minutes` without requests, after `max_hours`, or on POST /admin/shutdown.
"""

import asyncio
import hashlib
import json
import os
import re
import shutil
import subprocess
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
    "tunnel": {"kind": "cloudflared"},
    "idle_minutes": 30,
    "max_hours": 11.5,
    "replica_max_gb": 11.0,
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
STARTED = time.time()


def notify(event: str, **data):
    """Print a status event and publish it to the ntfy topic that `kis` follows."""
    msg = json.dumps({"event": event, "t": round(time.time() - STARTED), **data})
    print(msg, flush=True)
    if CONFIG["ntfy_topic"]:
        try:
            urllib.request.urlopen(f"https://ntfy.sh/{CONFIG['ntfy_topic']}", data=msg.encode(), timeout=10)
        except OSError as e:
            print("ntfy failed:", e, flush=True)


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


def launch(binary: str, model_path: str, draft_path: str | None, port: int, gpu_ids: list[str]) -> subprocess.Popen:
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
        str(MODEL["parallel"]),
    ]
    if draft_path:
        cmd += ["--model-draft", draft_path]
    if len(gpu_ids) > 1:
        cmd += ["--split-mode", "layer", "--tensor-split", ",".join("1" * len(gpu_ids))]
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


def start_backends(model_path: str, draft_path: str | None) -> dict[int, subprocess.Popen]:
    binary = llama_server_binary()
    gpus = subprocess.check_output(["nvidia-smi", "--query-gpu=index", "--format=csv,noheader"], text=True).split()
    size_gb = sum(os.path.getsize(p) for p in (model_path, draft_path) if p) / 1e9
    topology, groups = plan(gpus, size_gb, MODEL.get("topology"), CONFIG["replica_max_gb"])

    ports = range(BACKEND_PORT, BACKEND_PORT + len(groups))
    backends = {port: launch(binary, model_path, draft_path, port, g) for port, g in zip(ports, groups, strict=True)}
    for port, proc in backends.items():
        if not wait_healthy(port, proc):
            log = (WORK / f"llama-{port}.log").read_text()[-3000:]
            for other in backends.values():
                other.terminate()
            notify("error", error="llama-server failed to start (lower --parallel?)", log=log)
            raise RuntimeError("llama-server failed to start")
    notify("backends_ready", topology=topology, gpus=len(gpus), model_gb=round(size_gb, 2), parallel=MODEL["parallel"])
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


class Balancer:
    """Checks the API key and sends each request to the instance with the fewest in flight."""

    def __init__(self, ports: list[int], api_key: str):
        self.inflight = {port: 0 for port in ports}
        self.api_key = api_key
        self.last_activity = time.time()
        self.stop = asyncio.Event()

    @property
    def idle_seconds(self) -> float:
        return 0 if any(self.inflight.values()) else time.time() - self.last_activity

    def app(self) -> web.Application:
        async def client_session(app: web.Application):
            self.session = aiohttp.ClientSession(timeout=aiohttp.ClientTimeout(total=None), auto_decompress=False)
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
            self.stop.set()
            return web.json_response({"status": "stopping"})
        port = min(self.inflight, key=lambda p: self.inflight[p])
        self.inflight[port] += 1
        self.last_activity = time.time()
        try:
            return await self.forward(req, f"http://127.0.0.1:{port}{req.rel_url}")
        finally:
            self.inflight[port] -= 1
            self.last_activity = time.time()

    async def forward(self, req: web.Request, url: str) -> web.StreamResponse:
        """Proxy one request, streaming the response (SSE included) chunk by chunk."""
        headers = {k: v for k, v in req.headers.items() if k.lower() not in HOP_HEADERS}
        async with self.session.request(req.method, url, data=await req.read(), headers=headers) as upstream:
            resp = web.StreamResponse(
                status=upstream.status,
                headers={k: v for k, v in upstream.headers.items() if k.lower() not in HOP_HEADERS},
            )
            await resp.prepare(req)
            try:
                async for chunk in upstream.content.iter_any():
                    await resp.write(chunk)
                await resp.write_eof()
            except ConnectionResetError:  # client went away mid-stream; closing upstream stops generation
                pass
            return resp


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

    reason = "shutdown requested"
    while not balancer.stop.is_set():
        await asyncio.sleep(20)
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
