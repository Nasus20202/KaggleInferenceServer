"""Kaggle script: measure how much KV cache each model preset fits on the T4s.

For every model it finds
  context     the largest context per slot (>= MIN_CTX, <= the model's trained
              context) with CONTEXT_SLOTS slots in total
  throughput  the most slots per llama-server instance at MIN_CTX per slot
and reports the VRAM in use. Each attempt starts llama-server with the
preset's arguments, sends one request, and counts as a fit only if every GPU
keeps MARGIN_MIB free. Replica models are measured on one GPU (context on GPU
0, throughput on GPU 1, at the same time); split models on both.

Started by `kis calibrate`; results go to the ntfy topic as `calibrated` events.
"""

import json
import os
import shutil
import subprocess
import threading
import time
import urllib.request
from pathlib import Path

# <params> (replaced by `kis calibrate`)
RUN = ""
MODELS = {}
ARGS = []
NTFY_TOPIC = ""
REPLICA_MAX_GB = 11.0
CONTEXT_SLOTS = 4
MIN_CTX = 32768
# </params>

MAX_PARALLEL = 32
MARGIN_MIB = 512
CTX_STEPS = [262144, 196608, 163840, 131072, 114688, 98304, 81920, 65536, 49152, 40960, 32768]
MODELS_DIR = Path("/kaggle/tmp/models" if os.path.isdir("/kaggle/tmp") else "/tmp/models")
LOGS = Path("/kaggle/working")
STARTED = time.time()


def notify(event: str, **data):
    msg = json.dumps({"event": event, "t": round(time.time() - STARTED), "run": RUN, **data})
    print(msg, flush=True)
    if NTFY_TOPIC:
        try:
            urllib.request.urlopen(f"https://ntfy.sh/{NTFY_TOPIC}", data=msg.encode(), timeout=10)
        except OSError as e:
            print("ntfy failed:", e, flush=True)


def llama_server_binary() -> str:
    found = sorted(Path("/kaggle/input").rglob("llama-server"))
    if not found:
        raise RuntimeError("llama-server not found: attach the build kernel output (`kis build`)")
    shutil.copy(found[0], "/tmp/llama-server")
    os.chmod("/tmp/llama-server", 0o755)
    return "/tmp/llama-server"


def gpu_used_mib(gpus: list[str]) -> dict[str, tuple[int, int]]:
    out = subprocess.check_output(
        ["nvidia-smi", "--query-gpu=index,memory.used,memory.total", "--format=csv,noheader,nounits"], text=True
    )
    rows = {i: (int(u), int(t)) for i, u, t in (line.split(", ") for line in out.strip().splitlines())}
    return {g: rows[g] for g in gpus}


class Tester:
    """Runs fit attempts for one model on one GPU group."""

    def __init__(self, binary: str, model: dict, paths: tuple[str, str | None], gpus: list[str], port: int):
        self.binary, self.model, self.paths, self.gpus, self.port = binary, model, paths, gpus, port
        self.n_ctx_train = None

    def attempt(self, parallel: int, ctx: int) -> dict | None:
        """VRAM use if llama-server starts and serves with these settings, else None."""
        model_path, draft_path = self.paths
        cmd = [self.binary, "-m", model_path, "--host", "127.0.0.1", "--port", str(self.port)]
        cmd += ["--parallel", str(parallel), "--kv-unified-per-slot", str(ctx)]
        if draft_path:
            cmd += ["--model-draft", draft_path]
        if len(self.gpus) > 1:
            split = self.model.get("tensor_split") or ",".join("1" * len(self.gpus))
            cmd += ["--split-mode", "layer", "--tensor-split", split]
        cmd += ARGS + self.model.get("args", [])
        log_path = LOGS / f"calibrate-{self.port}.log"
        env = {**os.environ, "CUDA_VISIBLE_DEVICES": ",".join(self.gpus)}
        with open(log_path, "w") as log:
            proc = subprocess.Popen(cmd, env=env, stdout=log, stderr=subprocess.STDOUT)
        try:
            ok = self._serves(proc)
            used = gpu_used_mib(self.gpus) if ok else None
        finally:
            proc.terminate()
            try:
                proc.wait(30)
            except subprocess.TimeoutExpired:
                proc.kill()
                proc.wait()
        fits = bool(used) and all(total - u >= MARGIN_MIB for u, total in used.values())
        print(
            f"  gpus={','.join(self.gpus)} parallel={parallel} ctx={ctx}: {'fits' if fits else 'no'} {used}", flush=True
        )
        return {"used_mib": {g: u for g, (u, _) in used.items()}} if fits else None

    def _serves(self, proc: subprocess.Popen) -> bool:
        deadline = time.time() + 600
        while time.time() < deadline and proc.poll() is None:
            try:
                with urllib.request.urlopen(f"http://127.0.0.1:{self.port}/health", timeout=5):
                    break
            except OSError:
                time.sleep(2)
        else:
            return False
        if self.n_ctx_train is None:
            with urllib.request.urlopen(f"http://127.0.0.1:{self.port}/v1/models", timeout=30) as r:
                self.n_ctx_train = json.load(r)["data"][0]["meta"]["n_ctx_train"]
        body = json.dumps({"messages": [{"role": "user", "content": "Say hi."}], "max_tokens": 32}).encode()
        req = urllib.request.Request(
            f"http://127.0.0.1:{self.port}/v1/chat/completions", data=body, headers={"Content-Type": "application/json"}
        )
        try:
            with urllib.request.urlopen(req, timeout=300) as r:
                return r.status == 200
        except OSError:
            return False

    def max_ctx(self, parallel: int) -> dict:
        """Largest context per slot that fits `parallel` slots (descending steps)."""
        if self.n_ctx_train is None:  # one cheap start to read the trained context
            self.attempt(1, MIN_CTX)
        steps = [c for c in CTX_STEPS if c <= (self.n_ctx_train or CTX_STEPS[0])]
        for ctx in steps:
            if fit := self.attempt(parallel, ctx):
                return {"parallel": parallel, "ctx": ctx, **fit}
        return {"parallel": parallel, "ctx": None}

    def max_parallel(self, ctx: int) -> dict:
        """Most slots that fit at `ctx` per slot (binary search)."""
        lo, hi, best = 1, MAX_PARALLEL, None
        while lo <= hi:
            mid = (lo + hi) // 2
            if fit := self.attempt(mid, ctx):
                lo, best = mid + 1, {"parallel": mid, "ctx": ctx, **fit}
            else:
                hi = mid - 1
        return best or {"parallel": 0, "ctx": ctx}


def fetch(model: dict, file: str) -> str:
    from huggingface_hub import hf_hub_download

    os.environ.setdefault("HF_XET_HIGH_PERFORMANCE", "1")
    return hf_hub_download(model["repo"], file, revision=model.get("revision"), local_dir=MODELS_DIR)


def calibrate(binary: str, gpus: list[str], name: str, model: dict):
    model_path = fetch(model, model["file"])
    draft_path = fetch(model, model["draft_file"]) if model.get("draft_file") else None
    size_gb = sum(os.path.getsize(p) for p in (model_path, draft_path) if p) / 1e9
    topology = model.get("topology") or ("replicas" if size_gb <= REPLICA_MAX_GB else "split")
    paths = (model_path, draft_path)
    if topology == "replicas":
        instances = len(gpus)
        ctx_tester = Tester(binary, model, paths, [gpus[0]], 8090)
        par_tester = Tester(binary, model, paths, [gpus[-1]], 8091)
        result = {}
        thread = threading.Thread(target=lambda: result.update(throughput=par_tester.max_parallel(MIN_CTX)))
        thread.start()
        result["context"] = ctx_tester.max_ctx(max(1, CONTEXT_SLOTS // instances))
        thread.join()
        n_ctx_train = ctx_tester.n_ctx_train
    else:
        instances = 1
        tester = Tester(binary, model, paths, gpus, 8090)
        result = {"context": tester.max_ctx(CONTEXT_SLOTS), "throughput": tester.max_parallel(MIN_CTX)}
        n_ctx_train = tester.n_ctx_train
    notify(
        "calibrated",
        model=name,
        topology=topology,
        instances=instances,
        model_gb=round(size_gb, 2),
        n_ctx_train=n_ctx_train,
        **result,
    )
    for p in paths:
        if p:
            os.remove(p)


def main():
    binary = llama_server_binary()
    gpus = subprocess.check_output(["nvidia-smi", "--query-gpu=index", "--format=csv,noheader"], text=True).split()
    notify("calibrating", models=list(MODELS), gpus=gpu_used_mib(gpus))
    for name, model in MODELS.items():
        try:
            calibrate(binary, gpus, name, model)
        except Exception as e:  # keep going with the other models
            notify("calibrate_failed", model=name, error=repr(e)[:500])
    notify("calibration_done")


if __name__ == "__main__":
    try:
        main()
    except Exception as e:
        notify("error", error=repr(e)[:2000])
        raise
