"""Kaggle script: measure what each model preset fits on the T4s, and how fast it runs.

For every model and every profile (context per slot of 32K, 64K, 96K, 128K, or
"max" = the largest context that fits one slot, up to the model's trained context)
it finds the most slots that fit, then measures the speed of one request alone and
of all slots busy at once. Replica models are measured on one GPU per instance (both
GPUs work in parallel); if a profile doesn't fit one GPU, it is measured split over
both. Each attempt starts llama-server with the preset's arguments and sends a
request; it fits only if every GPU keeps MARGIN_MIB free.

Started by `kis calibrate`; results go to the ntfy topic as `calibrated` events.
"""

import bisect
import contextlib
import json
import os
import subprocess
import threading
import time
import urllib.request
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

from common import fetch, gpu_stats, llama_command, llama_server_binary, publish  # inlined by `kis`

# <params> (replaced by `kis calibrate`)
RUN = ""
MODELS = {}
ARGS = []
NTFY_TOPIC = ""
REPLICA_MAX_GB = 11.0
# </params>

PROFILES = {"32k": 32768, "64k": 65536, "96k": 98304, "128k": 131072}
CTX_STEP = 16384  # granularity of the "max" profile
MAX_PARALLEL = 32
MARGIN_MIB = 512
BENCH_TOKENS = 256
LOGS = Path("/kaggle/working")
STARTED = time.time()


def notify(event: str, **data):
    payload = {"event": event, "t": round(time.time() - STARTED), "run": RUN, **data}
    print(json.dumps(payload), flush=True)
    try:
        publish(NTFY_TOPIC, payload)
    except OSError as e:
        print("ntfy failed:", e, flush=True)


def gpu_used_mib(gpus: list[str]) -> dict[str, tuple[int, int]]:
    """(used, total) MiB of the given GPUs."""
    return {str(g["gpu"]): (g["used_mib"], g["total_mib"]) for g in gpu_stats() if str(g["gpu"]) in gpus}


def largest(values: list[int], fits, guess: int) -> tuple[int, dict] | None:
    """Largest value in ascending `values` for which fits(value) returns a result, starting at
    the value nearest `guess` and binary-searching up or down (fits is monotone)."""
    i = min(max(bisect.bisect_right(values, guess) - 1, 0), len(values) - 1)
    best = None
    if result := fits(values[i]):
        best, lo, hi = (values[i], result), i + 1, len(values) - 1
    else:
        lo, hi = 0, i - 1
    while lo <= hi:
        mid = (lo + hi) // 2
        if result := fits(values[mid]):
            best, lo = (values[mid], result), mid + 1
        else:
            hi = mid - 1
    return best


class Tester:
    """Starts llama-server for one model on one GPU group."""

    def __init__(self, binary: str, model: dict, paths: tuple[str, str | None], gpus: list[str], port: int):
        self.binary, self.model, self.paths, self.gpus, self.port = binary, model, paths, gpus, port
        self.n_ctx_train = None
        self.url = f"http://127.0.0.1:{port}"

    @contextlib.contextmanager
    def running(self, parallel: int, ctx: int):
        """Yield the VRAM use if llama-server starts, serves a request and leaves MARGIN_MIB free, else None."""
        cmd, env = llama_command(self.binary, self.model, self.paths, self.port, self.gpus, parallel, ctx, ARGS)
        with open(LOGS / f"calibrate-{self.port}.log", "w") as log:
            proc = subprocess.Popen(cmd, env=env, stdout=log, stderr=subprocess.STDOUT)
        try:
            used = gpu_used_mib(self.gpus) if self._serves(proc) else None
            if used is not None and len(used) < len(self.gpus):
                used = None  # nvidia-smi failed: can't tell whether it fits
            fits = bool(used) and all(total - u >= MARGIN_MIB for u, total in used.values())
            print(f"  gpus={','.join(self.gpus)} parallel={parallel} ctx={ctx}: {fits} {used}", flush=True)
            yield {"used_mib": {g: u for g, (u, _) in used.items()}} if fits else None
        finally:
            proc.terminate()
            try:
                proc.wait(30)
            except subprocess.TimeoutExpired:
                proc.kill()
                proc.wait()

    def _serves(self, proc: subprocess.Popen) -> bool:
        deadline = time.time() + 600
        while time.time() < deadline and proc.poll() is None:
            try:
                with urllib.request.urlopen(f"{self.url}/health", timeout=5):
                    break
            except OSError:
                time.sleep(2)
        else:
            return False
        if self.n_ctx_train is None:
            with urllib.request.urlopen(f"{self.url}/v1/models", timeout=30) as r:
                self.n_ctx_train = json.load(r)["data"][0]["meta"]["n_ctx_train"]
        try:
            self.complete(32)
            return True
        except OSError:
            return False

    def complete(self, max_tokens: int, ignore_eos: bool = False) -> int:
        body = {
            "messages": [{"role": "user", "content": "Write a detailed essay about the history of computing."}],
            "max_tokens": max_tokens,
            "ignore_eos": ignore_eos,
        }
        req = urllib.request.Request(
            f"{self.url}/v1/chat/completions",
            data=json.dumps(body).encode(),
            headers={"Content-Type": "application/json"},
        )
        with urllib.request.urlopen(req, timeout=900) as r:
            return json.load(r)["usage"]["completion_tokens"]

    def attempt(self, parallel: int, ctx: int) -> dict | None:
        with self.running(parallel, ctx) as fit:
            return fit

    def max_parallel(self, ctx: int, guess: int | None = None) -> dict:
        """Most slots that fit at `ctx` per slot."""
        found = largest(list(range(1, MAX_PARALLEL + 1)), lambda p: self.attempt(p, ctx), guess or MAX_PARALLEL // 2)
        return {"parallel": found[0], "ctx": ctx, **found[1]} if found else {"parallel": 0, "ctx": ctx}

    def max_ctx(self, guess: int) -> int | None:
        """Largest context (multiple of CTX_STEP, up to the trained context) that fits one slot."""
        steps = list(range(32768, self.n_ctx_train + 1, CTX_STEP))
        if self.n_ctx_train not in steps:
            steps.append(self.n_ctx_train)
        found = largest(steps, lambda c: self.attempt(1, c), guess)
        return found[0] if found else None

    def bench(self, parallel: int, ctx: int) -> dict:
        """Tokens per second of one request alone and of `parallel` concurrent requests."""
        with self.running(parallel, ctx) as fit:
            if not fit:
                return {}
            start = time.time()
            single = self.complete(BENCH_TOKENS, ignore_eos=True) / (time.time() - start)
            start = time.time()
            with ThreadPoolExecutor(parallel) as pool:
                tokens = sum(pool.map(lambda _: self.complete(BENCH_TOKENS, ignore_eos=True), range(parallel)))
            return {"single_tps": round(single, 1), "total_tps": round(tokens / (time.time() - start), 1)}


def in_parallel(*jobs):
    """Run jobs (callables) in threads, one per GPU, and return their results."""
    results = [None] * len(jobs)

    def run(i, job):
        results[i] = job()

    threads = [threading.Thread(target=run, args=(i, job)) for i, job in enumerate(jobs)]
    for t in threads:
        t.start()
    for t in threads:
        t.join()
    return results


def calibrate(binary: str, gpus: list[str], name: str, model: dict):
    paths = (fetch(model, model["file"]), fetch(model, model["draft_file"]) if model.get("draft_file") else None)
    size_gb = sum(os.path.getsize(p) for p in paths if p) / 1e9
    split = Tester(binary, model, paths, gpus, 8092)
    replicas = model.get("topology") != "split" and size_gb <= REPLICA_MAX_GB
    a, b = Tester(binary, model, paths, [gpus[0]], 8090), Tester(binary, model, paths, [gpus[-1]], 8091)
    main, instances = (a, len(gpus)) if replicas else (split, 1)

    # Most slots per profile: 32K and 64K searched from scratch, 96K and 128K from their estimates.
    if replicas:
        r32, r64 = in_parallel(lambda: a.max_parallel(32768), lambda: b.max_parallel(65536))
        b.n_ctx_train = b.n_ctx_train or a.n_ctx_train
    else:
        r32 = split.max_parallel(32768)
        r64 = split.max_parallel(65536, guess=r32["parallel"] // 2) if (split.n_ctx_train or 0) >= 65536 else None
    n_ctx_train = main.n_ctx_train
    if not n_ctx_train:
        raise RuntimeError("llama-server never started")
    kv_tokens = max(r32["parallel"] * 32768, (r64 or {}).get("parallel", 0) * 65536)
    found = {"32k": r32, "64k": r64}
    rest = [k for k in ("96k", "128k") if PROFILES[k] <= n_ctx_train]
    if replicas:
        jobs = [
            lambda k=k, t=t: t.max_parallel(PROFILES[k], guess=kv_tokens // PROFILES[k])
            for k, t in zip(rest, (a, b)[: len(rest)], strict=True)
        ]
        found.update(zip(rest, in_parallel(*jobs), strict=True))
    else:
        found.update({k: split.max_parallel(PROFILES[k], guess=kv_tokens // PROFILES[k]) for k in rest})

    profiles = {}
    for key, r in found.items():
        if r is None or PROFILES[key] > n_ctx_train:
            continue
        if r["parallel"]:
            profiles[key] = {"topology": "replicas" if replicas else "split", "instances": instances, **r}
        elif replicas:  # doesn't fit one GPU: split the model over both
            s = split.max_parallel(PROFILES[key], guess=1)
            if s["parallel"]:
                profiles[key] = {"topology": "split", "instances": 1, **s}

    # "max": the largest context that fits one slot, then as many slots as fit at it.
    ctx = main.max_ctx(guess=kv_tokens)
    topology, tester, n = ("replicas" if replicas else "split"), main, instances
    if replicas and (ctx or 0) < n_ctx_train:
        split.n_ctx_train = n_ctx_train
        split_ctx = split.max_ctx(guess=2 * kv_tokens)
        if (split_ctx or 0) > (ctx or 0):
            ctx, topology, tester, n = split_ctx, "split", split, 1
    if ctx:
        profiles["max"] = {"topology": topology, "instances": n, **tester.max_parallel(ctx, guess=1)}

    # Speed of each profile: replica benches alternate between the two GPUs.
    keys = list(profiles)
    testers = {"replicas": [a, b], "split": [split]}

    def bench(key: str, tester: Tester):
        p = profiles[key]
        speed = tester.bench(p["parallel"], p["ctx"])
        if "total_tps" in speed:
            speed["total_tps"] = round(speed["total_tps"] * p["instances"], 1)
        p.update(speed)

    for topo in ("replicas", "split"):
        todo = [k for k in keys if profiles[k]["topology"] == topo]
        pool = testers[topo]
        for i in range(0, len(todo), len(pool)):
            chunk = todo[i : i + len(pool)]
            in_parallel(*[lambda k=k, t=t: bench(k, t) for k, t in zip(chunk, pool[: len(chunk)], strict=True)])

    notify("calibrated", model=name, model_gb=round(size_gb, 2), n_ctx_train=n_ctx_train, profiles=profiles)
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
