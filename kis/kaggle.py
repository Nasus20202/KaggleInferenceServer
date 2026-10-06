"""Render the scripts in kaggle/ and run them on Kaggle through the kaggle CLI."""

import json
import pprint
import re
import subprocess
import sys

from .settings import ROOT, STATE_DIR

BUILD = "kis-llamacpp-build"
SERVER = "kis-server"


def kernel_id(config: dict, slug: str) -> str:
    return f"{config['kaggle']['username']}/{slug}"


def kernel_url(config: dict, slug: str) -> str:
    return f"https://www.kaggle.com/code/{kernel_id(config, slug)}"


def render(script: str, params: dict) -> str:
    """kaggle/<script> with the block between `# <params>` and `# </params>` replaced by `params`."""
    source = (ROOT / "kaggle" / script).read_text()
    block = "".join(
        f"{name} = {pprint.pformat(value, sort_dicts=False, width=110)}\n" for name, value in params.items()
    )
    rendered, n = re.subn(
        r"(# <params>.*?\n).*?(# </params>)", lambda m: m.group(1) + block + m.group(2), source, count=1, flags=re.S
    )
    assert n == 1, f"no params block in {script}"
    return rendered


def push(config: dict, slug: str, script: str, params: dict, sources: list[str] | None = None):
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

    result = _cli("kernels", "push", "-p", str(folder), "--accelerator", config["kaggle"]["accelerator"])
    if result.returncode and "unrecognized arguments" in result.stderr:
        print("! kaggle CLI has no --accelerator; choose GPU T4 x2 for the kernel in the Kaggle UI")
        result = _cli("kernels", "push", "-p", str(folder))
    print((result.stdout + result.stderr).strip())
    if result.returncode or "error" in result.stdout.lower():
        sys.exit("kaggle push failed")


def status(config: dict, slug: str) -> str:
    """Kernel status line (running / complete / error / cancel...) or the CLI's error text."""
    result = _cli("kernels", "status", kernel_id(config, slug))
    return (result.stdout + result.stderr).strip()


def _cli(*args: str) -> subprocess.CompletedProcess:
    return subprocess.run(["kaggle", *args], text=True, capture_output=True)
