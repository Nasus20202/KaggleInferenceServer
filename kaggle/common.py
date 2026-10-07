"""Shared by the Kaggle scripts (server.py, calibrate.py).

A Kaggle script kernel runs a single file, so `kis` pastes this file (and
kis/schema.py) over each script's import of it when it pushes the script.
Calibration measures what the server will run because both build the
llama-server command here.
"""

import hashlib
import json
import logging
import os
import shutil
import subprocess
import urllib.request
from pathlib import Path

from kis.schema import Event, GpuStats, Model, Ntfy

MODELS_DIR = Path("/kaggle/tmp/models" if os.path.isdir("/kaggle/tmp") else "/tmp/models")


def publish(ntfy: Ntfy, topic: str, payload: Event | str, headers: dict[str, str] | None = None) -> str:
    """Send one message (an event as JSON, or text with ntfy headers like Title) to a topic
    on the ntfy server, if the topic is set; return it. Raises OSError on failure."""
    msg = payload if isinstance(payload, str) else json.dumps(payload.to_json())
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
    of `ctx` tokens each (KV pool = parallel x ctx): `args` for every model, then the preset's."""
    model_path, draft_path = paths
    cmd = [binary, "-m", model_path, "--host", "127.0.0.1", "--port", str(port), "--alias", model.alias]
    cmd += ["--parallel", str(parallel), "--kv-unified-per-slot", str(ctx)]
    if draft_path:
        cmd += ["--model-draft", draft_path]
    if len(gpus) > 1:
        cmd += ["--split-mode", "layer", "--tensor-split", model.tensor_split or ",".join("1" * len(gpus))]
    cmd += args + model.args
    return cmd, {**os.environ, "CUDA_VISIBLE_DEVICES": ",".join(gpus)}
