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
from collections.abc import Callable, Iterator
from concurrent.futures import ThreadPoolExecutor
from dataclasses import dataclass, field
from pathlib import Path
from typing import TypeVar

from common import fetch, gpu_stats, llama_command, llama_server_binary, publish  # inlined by `kis`
from kis.schema import (  # inlined by `kis`
    MAX_PROFILE,
    PROFILE_CTX,
    CalibrateConfig,
    Calibration,
    Event,
    EventType,
    Fit,
    GpuStats,
    Model,
    Topology,
    dump,
    load,
)

# <params> (replaced by `kis calibrate`)
CONFIG = {"run": "", "models": {}}
# </params>

SETTINGS = load(CalibrateConfig, CONFIG)  # checked and typed; see kis/schema.py
R = TypeVar("R")

PROFILES = PROFILE_CTX
CTX_STEP = 16384  # granularity of the "max" profile
MAX_PARALLEL = 32
MARGIN_MIB = 512
BENCH_TOKENS = 256
LOGS = Path("/kaggle/working")
STARTED = time.time()


def notify(kind: EventType, **data: object) -> None:
    event = Event(kind, t=round(time.time() - STARTED), run=SETTINGS.run, data=data)
    print(json.dumps(event.to_json()), flush=True)
    try:
        publish(SETTINGS.ntfy, SETTINGS.ntfy_topic, event)
    except OSError as e:
        print("ntfy failed:", e, flush=True)


def gpu_memory(gpus: list[str]) -> dict[str, GpuStats]:
    """Memory of the given GPUs, by index."""
    return {str(g.gpu): g for g in gpu_stats() if str(g.gpu) in gpus}


def largest(values: list[int], fits: Callable[[int], R | None], guess: int) -> tuple[int, R] | None:
    """Largest value in ascending `values` for which fits(value) is not None, starting at
    the value nearest `guess` and binary-searching up or down (fits is monotone)."""
    i = min(max(bisect.bisect_right(values, guess) - 1, 0), len(values) - 1)
    best: tuple[int, R] | None = None
    if (result := fits(values[i])) is not None:
        best, lo, hi = (values[i], result), i + 1, len(values) - 1
    else:
        lo, hi = 0, i - 1
    while lo <= hi:
        mid = (lo + hi) // 2
        if (result := fits(values[mid])) is not None:
            best, lo = (values[mid], result), mid + 1
        else:
            hi = mid - 1
    return best


@dataclass(frozen=True)
class Slots:
    """The most slots of `ctx` tokens that fit (0: none), and the VRAM they use."""

    parallel: int
    ctx: int
    used_mib: dict[str, int] = field(default_factory=dict)

    def fit(self, topology: Topology, instances: int) -> Fit:
        return Fit(topology=topology, instances=instances, parallel=self.parallel, ctx=self.ctx, used_mib=self.used_mib)


@dataclass(frozen=True)
class Speed:
    single_tps: float  # one request alone
    total_tps: float  # all slots of one instance busy


class Tester:
    """Starts llama-server for one model on one GPU group."""

    def __init__(self, binary: str, model: Model, paths: tuple[str, str | None], gpus: list[str], port: int) -> None:
        self.binary, self.model, self.paths, self.gpus, self.port = binary, model, paths, gpus, port
        self.n_ctx_train: int | None = None
        self.url = f"http://127.0.0.1:{port}"

    @contextlib.contextmanager
    def running(self, parallel: int, ctx: int) -> Iterator[dict[str, int] | None]:
        """Yield the VRAM used per GPU (MiB) if llama-server starts, serves a request and
        leaves MARGIN_MIB free on every GPU, else None."""
        cmd, env = llama_command(
            self.binary, self.model, self.paths, self.port, self.gpus, parallel, ctx, SETTINGS.args
        )
        with open(LOGS / f"calibrate-{self.port}.log", "w") as log:
            proc = subprocess.Popen(cmd, env=env, stdout=log, stderr=subprocess.STDOUT)
        try:
            memory = gpu_memory(self.gpus) if self._serves(proc) else {}
            # all GPUs reported (nvidia-smi may fail: then we can't tell) and each keeps the margin
            fits = len(memory) == len(self.gpus) and all(
                g.total_mib - g.used_mib >= MARGIN_MIB for g in memory.values()
            )
            used = {i: g.used_mib for i, g in memory.items()}
            print(f"  gpus={','.join(self.gpus)} parallel={parallel} ctx={ctx}: {fits} {used}", flush=True)
            yield used if fits else None
        finally:
            proc.terminate()
            try:
                proc.wait(30)
            except subprocess.TimeoutExpired:
                proc.kill()
                proc.wait()

    def _serves(self, proc: subprocess.Popen[bytes]) -> bool:
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

    def attempt(self, parallel: int, ctx: int) -> dict[str, int] | None:
        with self.running(parallel, ctx) as fit:
            return fit

    def max_parallel(self, ctx: int, guess: int | None = None) -> Slots:
        """Most slots that fit at `ctx` per slot (0 if none)."""
        found = largest(list(range(1, MAX_PARALLEL + 1)), lambda p: self.attempt(p, ctx), guess or MAX_PARALLEL // 2)
        return Slots(found[0], ctx, found[1]) if found else Slots(0, ctx)

    def max_ctx(self, guess: int) -> int | None:
        """Largest context (multiple of CTX_STEP, up to the trained context) that fits one slot."""
        n_ctx_train = self.n_ctx_train or 32768
        steps = list(range(32768, n_ctx_train + 1, CTX_STEP))
        if n_ctx_train not in steps:
            steps.append(n_ctx_train)
        found = largest(steps, lambda c: self.attempt(1, c), guess)
        return found[0] if found else None

    def bench(self, parallel: int, ctx: int) -> Speed | None:
        """Tokens per second of one request alone and of `parallel` concurrent requests."""
        with self.running(parallel, ctx) as fit:
            if fit is None:
                return None
            start = time.time()
            single = self.complete(BENCH_TOKENS, ignore_eos=True) / (time.time() - start)
            start = time.time()
            with ThreadPoolExecutor(parallel) as pool:
                tokens = sum(pool.map(lambda _: self.complete(BENCH_TOKENS, ignore_eos=True), range(parallel)))
            return Speed(single_tps=round(single, 1), total_tps=round(tokens / (time.time() - start), 1))


def in_parallel(*jobs: Callable[[], R]) -> list[R]:
    """Run jobs (callables) in threads, one per GPU, and return their results."""
    results: list[R | None] = [None] * len(jobs)

    def run(i: int, job: Callable[[], R]) -> None:
        results[i] = job()

    threads = [threading.Thread(target=run, args=(i, job)) for i, job in enumerate(jobs)]
    for t in threads:
        t.start()
    for t in threads:
        t.join()
    return results  # type: ignore[return-value]  (every job has run)


def calibrate(binary: str, gpus: list[str], name: str, model: Model) -> None:
    paths = (fetch(model, model.file), fetch(model, model.draft_file) if model.draft_file else None)
    size_gb = sum(os.path.getsize(p) for p in paths if p) / 1e9
    split = Tester(binary, model, paths, gpus, 8092)
    replicas = model.topology != Topology.SPLIT and size_gb <= SETTINGS.replica_max_gb
    a, b = Tester(binary, model, paths, [gpus[0]], 8090), Tester(binary, model, paths, [gpus[-1]], 8091)
    main, instances = (a, len(gpus)) if replicas else (split, 1)

    # Most slots per profile: the two smallest searched from scratch, the others from their estimates.
    (small, small_ctx), (next_, next_ctx), *larger = PROFILES.items()
    if replicas:
        r_small, r_next = in_parallel(lambda: a.max_parallel(small_ctx), lambda: b.max_parallel(next_ctx))
        b.n_ctx_train = b.n_ctx_train or a.n_ctx_train
    else:
        r_small = split.max_parallel(small_ctx)
        fits_next = (split.n_ctx_train or 0) >= next_ctx
        r_next = split.max_parallel(next_ctx, guess=r_small.parallel // 2) if fits_next else None
    n_ctx_train = main.n_ctx_train
    if not n_ctx_train:
        raise RuntimeError("llama-server never started")
    kv_tokens = max(r_small.parallel * small_ctx, r_next.parallel * next_ctx if r_next else 0)
    found: dict[str, Slots | None] = {small: r_small, next_: r_next}
    rest = [k for k, ctx in larger if ctx <= n_ctx_train]
    if replicas:
        jobs = [
            lambda k=k, t=t: t.max_parallel(PROFILES[k], guess=kv_tokens // PROFILES[k])
            for k, t in zip(rest, (a, b)[: len(rest)], strict=True)
        ]
        found.update(zip(rest, in_parallel(*jobs), strict=True))
    else:
        found.update({k: split.max_parallel(PROFILES[k], guess=kv_tokens // PROFILES[k]) for k in rest})

    topology = Topology.REPLICAS if replicas else Topology.SPLIT
    profiles: dict[str, Fit] = {}
    for key, r in found.items():
        if r is None or PROFILES[key] > n_ctx_train:
            continue
        if r.parallel:
            profiles[key] = r.fit(topology, instances)
        elif replicas:  # doesn't fit one GPU: split the model over both
            s = split.max_parallel(PROFILES[key], guess=1)
            if s.parallel:
                profiles[key] = s.fit(Topology.SPLIT, 1)

    # "max": the largest context that fits one slot, then as many slots as fit at it.
    ctx = main.max_ctx(guess=kv_tokens)
    tester, n = main, instances
    if replicas and (ctx or 0) < n_ctx_train:
        split.n_ctx_train = n_ctx_train
        split_ctx = split.max_ctx(guess=2 * kv_tokens)
        if (split_ctx or 0) > (ctx or 0):
            ctx, topology, tester, n = split_ctx, Topology.SPLIT, split, 1
    if ctx:
        profiles[MAX_PROFILE] = tester.max_parallel(ctx, guess=1).fit(topology, n)

    # Speed of each profile: replica benches alternate between the two GPUs.
    keys = list(profiles)
    testers = {Topology.REPLICAS: [a, b], Topology.SPLIT: [split]}

    def bench(key: str, tester: Tester) -> None:
        p = profiles[key]
        if speed := tester.bench(p.parallel, p.ctx):
            p.single_tps, p.total_tps = speed.single_tps, round(speed.total_tps * p.instances, 1)

    for topo in Topology:
        todo = [k for k in keys if profiles[k].topology == topo]
        pool = testers[topo]
        for i in range(0, len(todo), len(pool)):
            chunk = todo[i : i + len(pool)]
            in_parallel(*[lambda k=k, t=t: bench(k, t) for k, t in zip(chunk, pool[: len(chunk)], strict=True)])

    result = Calibration(model=name, model_gb=round(size_gb, 2), n_ctx_train=n_ctx_train, profiles=profiles)
    notify(EventType.CALIBRATED, **dump(result))
    for p in paths:
        if p:
            os.remove(p)


def main() -> None:
    binary = llama_server_binary()
    gpus = subprocess.check_output(["nvidia-smi", "--query-gpu=index", "--format=csv,noheader"], text=True).split()
    notify(EventType.CALIBRATING, models=list(SETTINGS.models), gpus=dump(list(gpu_memory(gpus).values())))
    for name, model in SETTINGS.models.items():
        try:
            calibrate(binary, gpus, name, model)
        except Exception as e:  # keep going with the other models
            notify(EventType.CALIBRATE_FAILED, model=name, error=repr(e)[:500])
    notify(EventType.CALIBRATION_DONE)


if __name__ == "__main__":
    try:
        main()
    except Exception as e:
        notify(EventType.ERROR, error=repr(e)[:2000])
        raise
