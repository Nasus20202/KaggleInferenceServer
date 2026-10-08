"""Shared by the Kaggle scripts (server.py, calibrate.py).

A Kaggle script kernel runs a single file, so `kis` pastes this file (and
kis/schema.py) over each script's import of it when it pushes the script.
Calibration measures what the server will run because both build the
llama-server command here.
"""

import dataclasses
import hashlib
import json
import logging
import os
import shutil
import subprocess
import threading
import time
import urllib.request
from pathlib import Path

from kis.schema import CpuStats, Event, GpuStats, Model, Ntfy, RamStats

CGROUP, PROC = Path("/sys/fs/cgroup"), Path("/proc")  # the container's memory accounting, and the processes
MODELS_DIR = Path("/kaggle/tmp/models" if os.path.isdir("/kaggle/tmp") else "/tmp/models")
NTFY_MESSAGE_BYTES = 4096  # ntfy.sh's limit; a longer message becomes an attachment, which kis can't read


def event_message(event: Event) -> str:
    """An event as JSON, indented for reading in the ntfy app, or compact if that is too long."""
    data = event.to_json()
    pretty = json.dumps(data, indent=2)
    return pretty if len(pretty.encode()) <= NTFY_MESSAGE_BYTES else json.dumps(data)


def publish(ntfy: Ntfy, topic: str, payload: Event | str, headers: dict[str, str] | None = None) -> str:
    """Send one message (an event as JSON, or text with ntfy headers like Title) to a topic
    on the ntfy server, if the topic is set; return it. Raises OSError on failure."""
    msg = payload if isinstance(payload, str) else event_message(payload)
    if topic:
        req = urllib.request.Request(ntfy.url(topic), data=msg.encode(), headers={**ntfy.headers(), **(headers or {})})
        urllib.request.urlopen(req, timeout=10)
    return msg


def llama_server_binary() -> str:
    """Copy llama-server from the attached build output (inputs are read-only)."""
    found = sorted(Path("/kaggle/input").rglob("llama-server"))
    if not found:
        raise RuntimeError("llama-server not found: attach the build kernel output (`kis build`)")
    shutil.copy(found[0], "/tmp/llama-server")
    os.chmod("/tmp/llama-server", 0o755)
    return "/tmp/llama-server"


def fetch(model: Model, file: str, sha256: str | None = None) -> str:
    """Download one file of a model preset from Hugging Face; check its sha256 if given."""
    from huggingface_hub import hf_hub_download  # pyright: ignore[reportMissingImports]  (on Kaggle)

    os.environ.setdefault("HF_XET_HIGH_PERFORMANCE", "1")
    path: str = hf_hub_download(model.repo, file, revision=model.revision, local_dir=MODELS_DIR)
    if sha256:
        digest = hashlib.sha256()
        with open(path, "rb") as f:
            for chunk in iter(lambda: f.read(1 << 24), b""):
                digest.update(chunk)
        if digest.hexdigest() != sha256:
            raise RuntimeError(f"sha256 mismatch for {file}")
    return path


def fit_message(event: Event) -> Event:
    """`event` with the start of its `log` text cut until its message fits one ntfy message."""
    while len(event_message(event).encode()) > NTFY_MESSAGE_BYTES and event.data.get("log"):
        log = event.data["log"]
        event = dataclasses.replace(event, data={**event.data, "log": log[len(log) // 4 + 1 :]})
    return event


def gpu_stats() -> list[GpuStats]:
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
        logging.getLogger("kis").warning("nvidia-smi failed: %s", e)
        return []
    stats = []
    for line in out.strip().splitlines():
        gpu, used, total, util = map(int, line.split(", "))
        stats.append(GpuStats(gpu=gpu, used_mib=used, total_mib=total, util_pct=util))
    return stats


MIB = 1 << 20


def _read(path: Path) -> str | None:
    try:
        return path.read_text()
    except OSError:
        return None


def _field(text: str | None, name: str) -> int | None:
    """The number after `name` in a "name value" line, or None."""
    for line in (text or "").splitlines():
        parts = line.split()
        if len(parts) > 1 and parts[0].rstrip(":") == name and parts[1].isdigit():
            return int(parts[1])
    return None


def host_memory() -> tuple[int, int] | None:
    """(used, total) MiB of the container: its cgroup (v2, then v1) if it has a limit, else the
    machine's /proc/meminfo. Used leaves out the reclaimable page cache, like `docker stats`."""
    meminfo = _read(PROC / "meminfo")
    machine = _field(meminfo, "MemTotal")
    available = _field(meminfo, "MemAvailable")
    candidates = [  # (usage file, limit file, memory.stat file, its inactive-file key); all in bytes
        (CGROUP / "memory.current", CGROUP / "memory.max", CGROUP / "memory.stat", "inactive_file"),
        (
            CGROUP / "memory/memory.usage_in_bytes",
            CGROUP / "memory/memory.limit_in_bytes",
            CGROUP / "memory/memory.stat",
            "total_inactive_file",
        ),
    ]
    for usage_file, limit_file, stat_file, inactive_key in candidates:
        usage = _read(usage_file)
        if usage is None or not usage.strip().isdigit():
            continue
        used = int(usage) - (_field(_read(stat_file), inactive_key) or 0)
        limit = (_read(limit_file) or "").strip()
        total = int(limit) // MIB if limit.isdigit() else None  # "max" = unlimited
        if machine and (total is None or total > machine // 1024):
            total = machine // 1024
        if total:
            return max(used, 0) // MIB, total
    if machine and available is not None:
        return (machine - available) // 1024, machine // 1024
    return None


def process_memory(pid: int) -> int | None:
    """Anonymous resident memory (MiB) of a process: the heap and the prompt cache, not the
    memory-mapped model file; None if it is gone."""
    status = _read(PROC / str(pid) / "status")
    kib = _field(status, "RssAnon")
    return None if kib is None else kib // 1024


def ram_stats(pids: dict[int, int]) -> RamStats | None:
    """Host memory plus the anonymous memory of the llama-servers `pids` (port -> pid)."""
    memory = host_memory()
    if memory is None:
        return None
    sizes = {str(port): process_memory(pid) for port, pid in pids.items()}
    return RamStats(memory[0], memory[1], {port: mib for port, mib in sizes.items() if mib is not None})


def cpu_seconds() -> float | None:
    """CPU time used by the whole container so far: its cgroup (v2, then v1), else the machine's."""
    usec = _field(_read(CGROUP / "cpu.stat"), "usage_usec")
    if usec is not None:
        return usec / 1e6
    nsec = (_read(CGROUP / "cpuacct/cpuacct.usage") or _read(CGROUP / "cpu/cpuacct.usage") or "").strip()
    if nsec.isdigit():
        return int(nsec) / 1e9
    parts = ((_read(PROC / "stat") or "").splitlines() or [""])[0].split()[1:]  # "cpu user nice system idle ..."
    busy = [int(p) for p in parts[:3] + parts[5:8] if p.isdigit()]  # not idle, iowait or guest
    return sum(busy) / os.sysconf("SC_CLK_TCK") if busy else None


def cpu_cores() -> float:
    """CPUs the container may use: its cgroup quota if it has one, else the machine's."""
    quota = (_read(CGROUP / "cpu.max") or "").split()  # "max 100000" or "400000 100000"
    if len(quota) == 2 and quota[0].isdigit() and quota[1].isdigit() and int(quota[1]):
        return min(int(quota[0]) / int(quota[1]), float(os.cpu_count() or 1))
    return float(os.cpu_count() or 1)


def process_cpu_seconds(pid: int) -> float | None:
    """User + system CPU time of a process; None if it is gone."""
    stat = _read(PROC / str(pid) / "stat")
    if stat is None:
        return None
    fields = stat.rpartition(")")[2].split()  # after the command name, which may hold spaces
    if len(fields) < 13 or not (fields[11].isdigit() and fields[12].isdigit()):
        return None
    return (int(fields[11]) + int(fields[12])) / os.sysconf("SC_CLK_TCK")


class CpuMeter:
    """CPU use between one reading and the next. Readings less than MIN_WINDOW_S apart (the
    stats heartbeat right after the sampler) repeat the last result instead of a noisy one."""

    MIN_WINDOW_S = 5.0

    def __init__(self) -> None:
        self.lock = threading.Lock()
        self.last: tuple[float, float, dict[int, float]] | None = None  # time, container cpu s, per pid
        self.result: CpuStats | None = None

    def read(self, pids: dict[int, int]) -> CpuStats | None:
        now, total = time.monotonic(), cpu_seconds()
        if total is None:
            return None
        procs = {pid: s for pid in pids.values() if (s := process_cpu_seconds(pid)) is not None}
        with self.lock:
            if self.last is None or now - self.last[0] >= self.MIN_WINDOW_S:
                if self.last is not None:
                    before, total_before, procs_before = self.last
                    window, cores = now - before, cpu_cores()
                    used = (total - total_before) / window / cores * 100
                    by_port = {
                        str(port): round((procs[pid] - procs_before[pid]) / window * 100)
                        for port, pid in pids.items()
                        if pid in procs and pid in procs_before
                    }
                    self.result = CpuStats(cores, min(100, max(0, round(used))), by_port)
                self.last = (now, total, procs)
            return self.result


CPU = CpuMeter()


def cpu_stats(pids: dict[int, int]) -> CpuStats | None:
    """CPU use of the container and of the llama-servers `pids` (port -> pid) since the last call;
    None until there are two readings."""
    return CPU.read(pids)


def llama_command(
    binary: str,
    model: Model,
    paths: tuple[str, str | None],
    port: int,
    gpus: list[str],
    parallel: int,
    ctx: int,
    args: list[str],
) -> tuple[list[str], dict[str, str]]:
    """Command and environment of one llama-server instance on `gpus` with `parallel` slots
    of `ctx` tokens each (KV pool = parallel x ctx): `args` for every model, then the preset's.
    No web UI: the balancer requires an API key on every route, which a browser doesn't send."""
    model_path, draft_path = paths
    cmd = [binary, "-m", model_path, "--host", "127.0.0.1", "--port", str(port), "--alias", model.alias, "--no-webui"]
    cmd += ["--parallel", str(parallel), "--kv-unified-per-slot", str(ctx)]
    if draft_path:
        cmd += ["--model-draft", draft_path]
    if len(gpus) > 1:
        cmd += ["--split-mode", "layer", "--tensor-split", model.tensor_split or ",".join("1" * len(gpus))]
    cmd += args + model.args
    return cmd, {**os.environ, "CUDA_VISIBLE_DEVICES": ",".join(gpus)}
