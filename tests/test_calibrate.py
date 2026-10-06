"""kaggle/calibrate.py search logic, with llama-server attempts faked by a VRAM budget."""

import importlib.util
from pathlib import Path

import pytest

from kis import kaggle


@pytest.fixture
def calibrate():
    spec = importlib.util.spec_from_file_location(
        "kaggle_calibrate", Path(__file__).parent.parent / "kaggle" / "calibrate.py"
    )
    assert spec and spec.loader
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def fake_tester(calibrate, budget: int, n_ctx_train: int = 131072):
    """A Tester whose attempts fit while parallel x ctx <= budget tokens."""
    t = calibrate.Tester("llama-server", {}, ("m.gguf", None), ["0"], 8090)
    t.n_ctx_train = n_ctx_train
    t.attempt = lambda parallel, ctx: {"used_mib": {}} if parallel * ctx <= budget else None
    return t


def test_max_ctx_is_capped_by_the_trained_context(calibrate):
    assert fake_tester(calibrate, budget=4 * 262144, n_ctx_train=131072).max_ctx(4)["ctx"] == 131072
    assert fake_tester(calibrate, budget=4 * 70000).max_ctx(4)["ctx"] == 65536
    assert fake_tester(calibrate, budget=4 * 30000).max_ctx(4)["ctx"] is None


def test_max_parallel_finds_the_largest_fit(calibrate):
    assert fake_tester(calibrate, budget=7 * 32768).max_parallel(32768)["parallel"] == 7
    assert fake_tester(calibrate, budget=100 * 32768).max_parallel(32768)["parallel"] == calibrate.MAX_PARALLEL
    assert fake_tester(calibrate, budget=1000).max_parallel(32768)["parallel"] == 0


def test_render_keeps_the_constants_outside_params():
    source = kaggle.render("calibrate.py", {"MODELS": {}, "ARGS": [], "NTFY_TOPIC": "t"})
    namespace = {"__name__": "calibrate"}
    exec(source, namespace)
    assert namespace["MARGIN_MIB"] and namespace["MAX_PARALLEL"]
