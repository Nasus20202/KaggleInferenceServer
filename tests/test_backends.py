import dataclasses

import pytest


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
def backends(server, monkeypatch, tmp_path):
    """start_backends with fake processes: a launch fails with `log` while parallel > `fits`."""
    state = {"fits": 0, "log": "cudaMalloc failed: out of memory", "launched": [], "events": []}

    def launch(binary, model, paths, port, gpu_ids, parallel, ctx):
        state["launched"].append((parallel, ctx))
        (tmp_path / f"llama-{port}.log").write_text(state["log"] if parallel > state["fits"] else "ok")
        return FakeProc(parallel)

    monkeypatch.setattr(server, "WORK", tmp_path)
    monkeypatch.setattr(server, "server_binary", lambda: "llama-server")
    monkeypatch.setattr(server, "model_files", lambda model: ("model.gguf", None))
    monkeypatch.setattr(server.subprocess, "check_output", lambda *a, **k: "0\n1\n")
    monkeypatch.setattr(server.os.path, "getsize", lambda p: 3e9)
    monkeypatch.setattr(server, "gpu_stats", list)
    monkeypatch.setattr(server, "launch", launch)
    monkeypatch.setattr(server, "wait_healthy", lambda port, proc: proc.parallel <= state["fits"])
    monkeypatch.setattr(server, "notify", lambda event, **data: state["events"].append((event, data)))
    monkeypatch.setattr(server, "PRESETS", {"m": dataclasses.replace(server.MODEL, parallel=8)})
    monkeypatch.setattr(server, "FITTED", {})
    monkeypatch.setattr(server, "RUNTIME", server.Runtime())
    monkeypatch.setattr(server.SETTINGS, "ctx", 32768)
    return state


def test_keeps_context_and_lowers_slots_until_it_fits(server, backends):
    backends["fits"] = 4
    ports = server.start_backends("m")
    assert len(ports) == 2
    assert [p for p, _ in backends["launched"]] == [8, 8, 6, 6, 5, 5, 4, 4]
    assert {ctx for _, ctx in backends["launched"]} == {32768}
    event, data = backends["events"][-1]
    assert event == server.EventType.BACKENDS_READY
    assert (data["slots"], data["parallel"], data["ctx"]) == (8, 4, 32768)
    assert (data["preset"], server.RUNTIME.preset) == ("m", "m")


def test_loading_a_preset_again_starts_with_the_slots_that_fit(server, backends):
    backends["fits"] = 4
    server.start_backends("m")
    backends["launched"].clear()
    server.start_backends("m")
    assert backends["launched"] == [(4, 32768), (4, 32768)]


def test_gives_up_on_other_errors(server, backends):
    backends["log"] = "unknown model architecture"
    with pytest.raises(server.BackendsFailed) as failed:
        server.start_backends("m")
    assert backends["launched"] == [(8, 32768), (8, 32768)]
    assert "unknown model architecture" in failed.value.log
    assert not backends["events"]  # the caller reports it: an error at startup, a failed swap later
