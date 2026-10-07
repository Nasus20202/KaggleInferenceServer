"""Shared by the Kaggle scripts (server.py, calibrate.py).

A Kaggle script kernel runs a single file, so `kis` pastes this file over each
script's import of it when it pushes the script. Calibration measures what the
server will run because both build the llama-server command here.
"""

import hashlib
import json
import logging
import os
import shutil
import subprocess
import urllib.request
from pathlib import Path

MODELS_DIR = Path("/kaggle/tmp/models" if os.path.isdir("/kaggle/tmp") else "/tmp/models")


def publish(topic: str, payload: dict) -> str:
    """Send one JSON event to an ntfy.sh topic (if set); return it. Raises OSError on failure."""
    msg = json.dumps(payload)
    if topic:
        urllib.request.urlopen(f"https://ntfy.sh/{topic}", data=msg.encode(), timeout=10)
    return msg


def llama_server_binary() -> str:
    """Copy llama-server from the attached build output (inputs are read-only)."""
    found = sorted(Path("/kaggle/input").rglob("llama-server"))
    if not found:
        raise RuntimeError("llama-server not found: attach the build kernel output (`kis build`)")
    shutil.copy(found[0], "/tmp/llama-server")
    os.chmod("/tmp/llama-server", 0o755)
    return "/tmp/llama-server"


def fetch(model: dict, file: str, sha256: str | None = None) -> str:
    """Download one file of a model preset from Hugging Face; check its sha256 if given."""
    from huggingface_hub import hf_hub_download

    os.environ.setdefault("HF_XET_HIGH_PERFORMANCE", "1")
    path = hf_hub_download(model["repo"], file, revision=model.get("revision"), local_dir=MODELS_DIR)
    if sha256:
        digest = hashlib.sha256()
        with open(path, "rb") as f:
            for chunk in iter(lambda: f.read(1 << 24), b""):
                digest.update(chunk)
        if digest.hexdigest() != sha256:
            raise RuntimeError(f"sha256 mismatch for {file}")
    return path


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
        logging.getLogger("kis").warning("nvidia-smi failed: %s", e)
        return []
    keys = ("gpu", "used_mib", "total_mib", "util_pct")
    return [dict(zip(keys, map(int, line.split(", ")), strict=True)) for line in out.strip().splitlines()]


def llama_command(
    binary: str,
    model: dict,
    paths: tuple[str, str | None],
    port: int,
    gpus: list[str],
    parallel: int,
    ctx: int,
    args: list[str],
) -> tuple[list[str], dict]:
    """Command and environment of one llama-server instance on `gpus` with `parallel` slots
    of `ctx` tokens each (KV pool = parallel x ctx): `args` for every model, then the preset's."""
    model_path, draft_path = paths
    cmd = [binary, "-m", model_path, "--host", "127.0.0.1", "--port", str(port), "--alias", model["alias"]]
    cmd += ["--parallel", str(parallel), "--kv-unified-per-slot", str(ctx)]
    if draft_path:
        cmd += ["--model-draft", draft_path]
    if len(gpus) > 1:
        cmd += ["--split-mode", "layer", "--tensor-split", model.get("tensor_split") or ",".join("1" * len(gpus))]
    cmd += args + model.get("args", [])
    return cmd, {**os.environ, "CUDA_VISIBLE_DEVICES": ",".join(gpus)}
