import dataclasses

import pytest

import backends
from kis.schema import EventType, Topology
from state import Runtime


class FakeProc:
    def __init__(self, parallel: int):
        self.parallel = parallel

    def terminate(self):
        pass

    def wait(self):
        pass

    def poll(self):
        return None


@pytest.fixture
def fake(server, monkeypatch, tmp_path):
    """start_backends with fake processes: a launch fails with `log` while parallel > `fits`."""
    state = {"fits": 0, "log": "cudaMalloc failed: out of memory", "launched": [], "events": []}

    def launch(binary, model, paths, port, gpu_ids, parallel, ctx, args):
        state["launched"].append((parallel, ctx))
        (tmp_path / f"llama-{port}.log").write_text(state["log"] if parallel > state["fits"] else "ok")
        return FakeProc(parallel)

    monkeypatch.setattr(backends, "WORK", tmp_path)
    monkeypatch.setattr(backends, "server_binary", lambda: "llama-server")
    monkeypatch.setattr(backends, "model_files", lambda model: ("model.gguf", None))
    monkeypatch.setattr(backends.subprocess, "check_output", lambda *a, **k: "0\n1\n")
    monkeypatch.setattr(backends.os.path, "getsize", lambda p: 3e9)
    monkeypatch.setattr(backends, "gpu_stats", list)
    monkeypatch.setattr(backends, "launch", launch)
    monkeypatch.setattr(backends, "wait_healthy", lambda port, proc: proc.parallel <= state["fits"])
    monkeypatch.setattr(backends, "FITTED", {})
    monkeypatch.setattr(server.SETTINGS, "ctx", 32768)
    presets, runtime = {"m": dataclasses.replace(server.MODEL, parallel=8)}, Runtime()
    state["runtime"] = runtime

    def notify(event, **data):
        state["events"].append((event, data))

    state["start"] = lambda: backends.start_backends("m", presets, server.SETTINGS, runtime, notify)
    return state


def test_keeps_context_and_lowers_slots_until_it_fits(fake):
    fake["fits"] = 4
    ports = fake["start"]()
    assert len(ports) == 2
    assert [p for p, _ in fake["launched"]] == [8, 8, 6, 6, 5, 5, 4, 4]
    assert {ctx for _, ctx in fake["launched"]} == {32768}
    event, data = fake["events"][-1]
    assert event == EventType.BACKENDS_READY
    assert (data["slots"], data["parallel"], data["ctx"]) == (8, 4, 32768)
    assert (data["preset"], fake["runtime"].preset) == ("m", "m")


def test_loading_a_preset_again_starts_with_the_slots_that_fit(fake):
    fake["fits"] = 4
    fake["start"]()
    fake["launched"].clear()
    fake["start"]()
    assert fake["launched"] == [(4, 32768), (4, 32768)]


def test_gives_up_on_other_errors(fake):
    fake["log"] = "unknown model architecture"
    with pytest.raises(backends.BackendsFailed) as failed:
        fake["start"]()
    assert fake["launched"] == [(8, 32768), (8, 32768)]
    assert "unknown model architecture" in failed.value.log
    assert not fake["events"]  # the caller reports it: an error at startup, a failed swap later


def test_plan_replicates_small_models_and_splits_large():
    assert backends.plan(["0", "1"], 5.9, None, 11.0) == ("replicas", [["0"], ["1"]])
    assert backends.plan(["0", "1"], 17.1, None, 11.0) == ("split", [["0", "1"]])
    assert backends.plan(["0", "1"], 5.9, Topology.SPLIT, 11.0) == ("split", [["0", "1"]])
    assert backends.plan(["0"], 5.9, None, 11.0) == ("replicas", [["0"]])


def test_describe_exit():
    assert backends.describe_exit(1) == "exited with code 1"
    assert backends.describe_exit(-11) == "killed by SIGSEGV"
    assert "OOM killer" in backends.describe_exit(-9)


def test_crash_names_the_exited_process_and_keeps_the_log_tail(monkeypatch, tmp_path):
    class Exited(FakeProc):
        def __init__(self, code):
            super().__init__(1)
            self.code = code

        def poll(self):
            return self.code

    monkeypatch.setattr(backends, "WORK", tmp_path)
    (tmp_path / "llama-8091.log").write_text("x" * 5000 + "slot 3 released")
    live = backends.Backends.__new__(backends.Backends)
    live.procs = {8090: Exited(None), 8091: Exited(-9)}  # type: ignore[assignment]
    live.residents, live.resident_procs = {}, []
    crash = live.crash()
    assert crash and crash["crashed_port"] == 8091 and crash["exit_code"] == -9
    assert "OOM killer" in str(crash["exit"])
    assert str(crash["log"]).endswith("slot 3 released") and len(str(crash["log"])) == 3000
    live.procs = {8090: Exited(None)}  # type: ignore[assignment]
    assert live.crash() is None
