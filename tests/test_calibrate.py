"""kaggle/calibrate.py search logic, with llama-server attempts faked by a KV budget."""

import dataclasses
import importlib.util
from pathlib import Path

import pytest

from kis import kaggle
from kis.schema import Calibration, Model, Topology, load

MODEL = Model(repo="r", file="m.gguf", alias="m", parallel=1)


@pytest.fixture
def calibrate():
    spec = importlib.util.spec_from_file_location(
        "kaggle_calibrate", Path(__file__).parent.parent / "kaggle" / "calibrate.py"
    )
    assert spec and spec.loader
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def fake_tester(calibrate, budget: int, n_ctx_train: int = 262144):
    """A Tester whose attempts fit while parallel x ctx <= budget tokens; records attempts."""
    t = calibrate.Tester("llama-server", MODEL, ("m.gguf", None), ["0"], 8090)
    t.n_ctx_train, t.tries = n_ctx_train, []
    t.attempt = lambda parallel, ctx: (
        t.tries.append((parallel, ctx)) or ({"0": 1000} if parallel * ctx <= budget else None)
    )
    return t


@pytest.mark.parametrize("guess", [1, 5, 7, 8, 30])
def test_largest_finds_the_boundary_from_any_guess(calibrate, guess):
    found = calibrate.largest(list(range(1, 33)), lambda p: {"p": p} if p <= 7 else None, guess)
    assert found == (7, {"p": 7})
    assert calibrate.largest([1, 2], lambda p: None, guess) is None


def test_max_parallel(calibrate):
    assert fake_tester(calibrate, budget=7 * 32768).max_parallel(32768) == calibrate.Slots(7, 32768, {"0": 1000})
    assert fake_tester(calibrate, budget=100 * 32768).max_parallel(32768).parallel == calibrate.MAX_PARALLEL
    assert fake_tester(calibrate, budget=1000).max_parallel(32768) == calibrate.Slots(0, 32768)
    tester = fake_tester(calibrate, budget=3 * 65536)
    assert tester.max_parallel(65536, guess=3).parallel == 3
    assert len(tester.tries) <= 6  # a good guess needs few starts


def test_max_ctx_is_capped_by_the_trained_context(calibrate):
    assert fake_tester(calibrate, budget=10**7, n_ctx_train=131072).max_ctx(guess=10**7) == 131072
    assert fake_tester(calibrate, budget=200000).max_ctx(guess=150000) == 196608
    assert fake_tester(calibrate, budget=20000).max_ctx(guess=65536) is None


def test_render_keeps_the_constants_outside_params():
    source = kaggle.render("calibrate.py", {"CONFIG": {"run": "1", "models": {}, "ntfy_topic": "t"}})
    namespace = {"__name__": "calibrate"}
    exec(source, namespace)
    assert namespace["MARGIN_MIB"] and namespace["MAX_PARALLEL"]


def run_calibrate(calibrate, monkeypatch, size_gb: float, budget_per_gpu: int, n_ctx_train: int, model=None):
    """calibrate() for one model on 2 fake GPUs, each holding `budget_per_gpu` KV tokens."""
    events = []

    def attempt(self, parallel, ctx):
        self.n_ctx_train = n_ctx_train
        return {"0": 1000} if parallel * ctx <= budget_per_gpu * len(self.gpus) else None

    monkeypatch.setattr(calibrate.Tester, "attempt", attempt)
    monkeypatch.setattr(calibrate.Tester, "bench", lambda self, p, c: calibrate.Speed(40.0, 10.0 * p))
    monkeypatch.setattr(calibrate, "fetch", lambda model, file: file)
    monkeypatch.setattr(calibrate.os.path, "getsize", lambda p: size_gb * 1e9)
    monkeypatch.setattr(calibrate.os, "remove", lambda p: None)
    monkeypatch.setattr(calibrate, "notify", lambda event, **data: events.append(data))
    calibrate.calibrate("llama-server", ["0", "1"], "m", model or MODEL)
    return load(Calibration, events[-1]).profiles


def test_calibrate_small_model_fills_context_then_slots(calibrate, monkeypatch):
    profiles = run_calibrate(calibrate, monkeypatch, size_gb=3, budget_per_gpu=300000, n_ctx_train=262144)
    assert {k: (p.topology, p.parallel, p.ctx) for k, p in profiles.items()} == {
        "32k": ("replicas", 9, 32768),
        "64k": ("replicas", 4, 65536),
        "96k": ("replicas", 3, 98304),
        "128k": ("replicas", 2, 131072),
        "max": ("replicas", 1, 262144),  # capped by the trained context
    }
    assert profiles["32k"].total_tps == 180.0  # 2 replicas x 9 slots x 10


def test_calibrate_splits_what_does_not_fit_one_gpu(calibrate, monkeypatch):
    profiles = run_calibrate(calibrate, monkeypatch, size_gb=5, budget_per_gpu=100000, n_ctx_train=262144)
    assert profiles["64k"].topology == Topology.REPLICAS
    assert (profiles["128k"].topology, profiles["128k"].parallel) == (Topology.SPLIT, 1)
    assert (profiles["max"].topology, profiles["max"].ctx) == (Topology.SPLIT, 196608)


def test_calibrate_split_model(calibrate, monkeypatch):
    model = dataclasses.replace(MODEL, topology=Topology.SPLIT)
    profiles = run_calibrate(calibrate, monkeypatch, size_gb=17, budget_per_gpu=80000, n_ctx_train=262144, model=model)
    assert {k: (p.parallel, p.ctx) for k, p in profiles.items()} == {
        "32k": (4, 32768),
        "64k": (2, 65536),
        "96k": (1, 98304),
        "128k": (1, 131072),
        "max": (1, 147456),
    }
