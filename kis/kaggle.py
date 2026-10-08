"""Render the scripts in kaggle/ and run them on Kaggle through the kaggle CLI."""

import enum
import json
import pprint
import re
import subprocess
import time
from collections.abc import Mapping

from . import retry
from .schema import dump
from .settings import ROOT, STATE_DIR, Config

# kernel slugs and the script each one runs
BUILD, BUILD_SCRIPT = "kis-llamacpp-build", "build.py"
SERVER, SERVER_SCRIPT = "kis-server", "server.py"
CALIBRATE, CALIBRATE_SCRIPT = "kis-calibrate", "calibrate.py"
CONFIG_PARAM = "CONFIG"  # the params variable of server.py and calibrate.py
LLAMA_CPP_REF_PARAM = "LLAMA_CPP_REF"  # the params variable of build.py


class KernelState(enum.StrEnum):
    """Kaggle's KernelWorkerStatus, lower case."""

    QUEUED = "queued"
    RUNNING = "running"
    COMPLETE = "complete"
    ERROR = "error"
    CANCEL_REQUESTED = "cancel_requested"
    CANCEL_ACKNOWLEDGED = "cancel_acknowledged"
    UNKNOWN = "-"  # no such kernel (yet), or the API failed

    @property
    def ended(self) -> bool:
        return self in (self.COMPLETE, self.ERROR, self.CANCEL_REQUESTED, self.CANCEL_ACKNOWLEDGED)


# Modules a Kaggle script imports, pasted into it when it is pushed (Kaggle runs a single file).
INLINED = {
    "common": ROOT / "kaggle" / "common.py",
    "kis.schema": ROOT / "kis" / "schema.py",
    "state": ROOT / "kaggle" / "state.py",
    "tunnels": ROOT / "kaggle" / "tunnels.py",
}
INLINED_IMPORT = re.compile(r"^from (" + "|".join(map(re.escape, INLINED)) + r") import (?:\([^)]*\)|.*)$", re.M)


class KaggleError(RuntimeError):
    pass


def kernel_id(config: Config, slug: str) -> str:
    return f"{config.kaggle.username}/{slug}"


def kernel_url(config: Config, slug: str) -> str:
    return f"https://www.kaggle.com/code/{kernel_id(config, slug)}"


def build_source(config: Config) -> str:
    """The build kernel, attached to server and calibration kernels for its llama-server."""
    return kernel_id(config, BUILD)


def render(script: str, params: Mapping[str, object]) -> str:
    """kaggle/<script> with the block between `# <params>` and `# </params>` replaced by
    `params` (dataclasses as plain dicts), and each import of an INLINED module replaced
    by its source, the first time it appears."""
    source = (ROOT / "kaggle" / script).read_text()
    block = "".join(
        f"{name} = {pprint.pformat(dump(value), sort_dicts=False, width=110)}\n" for name, value in params.items()
    )
    rendered, n = re.subn(
        r"(# <params>.*?\n).*?(# </params>)", lambda m: m.group(1) + block + m.group(2), source, count=1, flags=re.S
    )
    assert n == 1, f"no params block in {script}"
    done: set[str] = set()
    while m := INLINED_IMPORT.search(rendered):
        module = m.group(1)
        pasted = "" if module in done else f"# --- {module}\n{INLINED[module].read_text()}# ---"
        done.add(module)
        rendered = rendered[: m.start()] + pasted + rendered[m.end() :]
    return rendered


def push(
    config: Config, slug: str, script: str, params: Mapping[str, object], sources: list[str] | None = None
) -> None:
    """Upload a new kernel version; Kaggle runs it in the background with GPU and internet."""
    folder = STATE_DIR / slug
    folder.mkdir(parents=True, exist_ok=True)
    (folder / script).write_text(render(script, params))
    meta = {
        "id": kernel_id(config, slug),
        "title": slug,
        "code_file": script,
        "language": "python",
        "kernel_type": "script",
        "is_private": True,
        "enable_gpu": True,
        "enable_tpu": False,
        "enable_internet": True,
        "dataset_sources": [],
        "competition_sources": [],
        "model_sources": [],
        "kernel_sources": sources or [],
    }
    (folder / "kernel-metadata.json").write_text(json.dumps(meta, indent=2))

    result = _cli("kernels", "push", "-p", str(folder), "--accelerator", config.kaggle.accelerator)
    if result.returncode and "unrecognized arguments" in result.stderr:
        print("! kaggle CLI has no --accelerator; choose GPU T4 x2 for the kernel in the Kaggle UI")
        result = _cli("kernels", "push", "-p", str(folder))
    print((result.stdout + result.stderr).strip())
    if result.returncode or "error" in result.stdout.lower():
        raise KaggleError(f"kaggle push of {slug} failed")


def status(config: Config, slug: str) -> str:
    """Kernel status line (running / complete / error / cancel...) or the CLI's error text."""
    attempt = 0
    while True:
        result = _cli("kernels", "status", kernel_id(config, slug))
        output = (result.stdout + result.stderr).strip()
        permanent = "Cannot access kernel" in output  # e.g. not created yet
        if result.returncode == 0 or permanent or attempt == retry.ATTEMPTS - 1:
            return output
        time.sleep(retry.backoff(attempt))  # Kaggle API hiccup
        attempt += 1


def state(config: Config, slug: str) -> KernelState:
    m = re.search(r'has status "(?:KernelWorkerStatus\.)?(\w+)"', status(config, slug))
    try:
        return KernelState(m.group(1).lower()) if m else KernelState.UNKNOWN
    except ValueError:
        return KernelState.UNKNOWN


def _cli(*args: str) -> subprocess.CompletedProcess[str]:
    return subprocess.run(["kaggle", *args], text=True, capture_output=True)
