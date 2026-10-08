"""llama-server instances: download the presets, start the instances of the loaded preset
(and of the resident ones) and stop them.

Inlined into server.py by `kis` (see state.py). The session's settings, the presets, the
runtime layout and `notify` come from server.py as arguments.
"""

import functools
import os
import re
import signal
import subprocess
import threading
import time
import urllib.request
from collections.abc import Callable

from common import fetch, gpu_stats, llama_command, llama_server_binary  # inlined by `kis`
from kis.schema import EventType, Model, ServerConfig, Topology, dump  # inlined by `kis`
from state import WORK, Runtime, log, log_file  # inlined by `kis`

BACKEND_PORT = 8090  # llama-server instances use BACKEND_PORT, BACKEND_PORT + 1, ...
RESIDENT_PORT = 8070  # resident presets use RESIDENT_PORT, RESIDENT_PORT + 1, ...

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


def prefetch(names: list[str], presets: dict[str, Model]) -> None:
    """Download presets in the background, so swapping to them only has to load them."""
    for name in names:
        try:
            model_files(presets[name])
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
    args: list[str],
) -> subprocess.Popen[bytes]:
    """One llama-server with `parallel` slots of `ctx` tokens each (KV pool = parallel x ctx)."""
    cmd, env = llama_command(binary, model, paths, port, gpu_ids, parallel, ctx, args)
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


CRASH_LOG_CHARS = 3000  # of a crashed llama-server's log; notify() cuts it further to fit a ntfy message


def describe_exit(code: int) -> str:
    """How a process ended, from its return code (negative: the signal that killed it)."""
    if code >= 0:
        return f"exited with code {code}"
    name = signal.Signals(-code).name if -code in signal.valid_signals() else f"signal {-code}"
    hint = " (the kernel's OOM killer sends it when the container is out of RAM)" if -code == signal.SIGKILL else ""
    return f"killed by {name}{hint}"


class BackendsFailed(RuntimeError):
    """llama-server didn't start; `log` is the tail of its output."""

    def __init__(self, message: str, log: str) -> None:
        super().__init__(message)
        self.log = log


def start_backends(
    name: str,
    presets: dict[str, Model],
    settings: ServerConfig,
    runtime: Runtime,
    notify: Callable[..., None],
) -> dict[int, subprocess.Popen[bytes]]:
    """Start one llama-server per GPU group for preset `name`. The context per slot is
    fixed; if the KV pool doesn't fit, retry with fewer slots. Record the layout in `runtime`."""
    model = presets[name]
    paths = model_files(model)
    binary = server_binary()
    gpus = subprocess.check_output(["nvidia-smi", "--query-gpu=index", "--format=csv,noheader"], text=True).split()
    size_gb = sum(os.path.getsize(p) for p in paths if p) / 1e9
    topology, groups = plan(gpus, size_gb, model.topology, settings.replica_max_gb)
    parallel, ctx = FITTED.get(name, model.parallel), model.ctx or settings.ctx

    ports = range(BACKEND_PORT, BACKEND_PORT + len(groups))
    while True:
        backends = {
            port: launch(binary, model, paths, port, g, parallel, ctx, settings.args)
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
            notify(EventType.RETRY, reason="out of memory", model=model.alias, parallel=parallel, ctx=ctx)
            continue
        raise BackendsFailed(f"llama-server failed to start with {name} (lower --ctx or --parallel?)", tail)
    FITTED[name] = parallel
    runtime.preset, runtime.model, runtime.topology = name, model.alias, topology
    runtime.slots, runtime.parallel, runtime.ctx = parallel * len(groups), parallel, ctx
    notify(EventType.BACKENDS_READY, model_gb=round(size_gb, 2), **dump(runtime), gpus=dump(gpu_stats()))
    return backends


def start_resident(name: str, port: int, presets: dict[str, Model], settings: ServerConfig) -> subprocess.Popen[bytes]:
    """One llama-server on GPU 0 for resident preset `name`, kept for the whole session."""
    model = presets[name]
    proc = launch(
        server_binary(),
        model,
        model_files(model),
        port,
        ["0"],
        model.parallel,
        model.ctx or settings.ctx,
        settings.args,
    )
    if not wait_healthy(port, proc):
        proc.terminate()
        proc.wait()
        tail = (WORK / f"llama-{port}.log").read_text()[-3000:]
        raise BackendsFailed(f"llama-server failed to start with resident {name}", tail)
    log.info("resident %s ready on :%d", name, port)
    return proc


class Backends:
    """The llama-server instances of the loaded preset and of the resident presets.
    Blocking: the balancer calls fetch and load in a thread."""

    def __init__(
        self, settings: ServerConfig, presets: dict[str, Model], runtime: Runtime, notify: Callable[..., None]
    ) -> None:
        self.settings, self.presets, self.runtime, self.notify = settings, presets, runtime, notify
        self.procs: dict[int, subprocess.Popen[bytes]] = {}
        self.residents: dict[str, int] = {}  # resident preset -> port
        self.resident_procs: list[subprocess.Popen[bytes]] = []

    def start_residents(self, names: list[str]) -> None:
        """Start the resident presets; before the swapped one, so its out-of-memory retries
        leave room for them."""
        for port, name in enumerate(names, RESIDENT_PORT):
            self.resident_procs.append(start_resident(name, port, self.presets, self.settings))
            self.residents[name] = port

    @property
    def ports(self) -> list[int]:
        return list(self.procs)

    def fetch(self, name: str) -> None:
        model_files(self.presets[name])

    def downloaded(self, name: str) -> bool:
        return downloaded(self.presets[name])

    def load(self, name: str) -> list[int]:
        """Replace the running instances with preset `name`'s; return their ports."""
        self.stop()
        self.procs = start_backends(name, self.presets, self.settings, self.runtime, self.notify)
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

    @property
    def pids(self) -> dict[int, int]:
        """Port -> pid of every running llama-server, resident ones too."""
        residents = zip(self.residents.values(), self.resident_procs, strict=False)
        return {port: proc.pid for port, proc in [*self.procs.items(), *residents]}

    def crash(self) -> dict[str, object] | None:
        """Stopped-event fields for the first llama-server that has exited: how it ended and its log tail."""
        residents = zip(self.residents.values(), self.resident_procs, strict=False)
        for port, proc in [*self.procs.items(), *residents]:
            code = proc.poll()
            if code is not None:
                try:
                    tail = (WORK / f"llama-{port}.log").read_text(errors="replace")[-CRASH_LOG_CHARS:]
                except OSError:
                    tail = ""
                return {"crashed_port": port, "exit_code": code, "exit": describe_exit(code), "log": tail}
        return None

    def close(self) -> None:
        self.stop()
        for proc in self.resident_procs:
            proc.terminate()
            proc.wait()
