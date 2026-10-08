"""kaggle/common.py: the llama-server command shared by the server and calibration."""

import dataclasses
import json

import common
from kis.schema import CpuStats, Event, EventType, GpuStats, Model, Ntfy, RamStats, Stats, Topology, UsageSummary, dump

MODEL = Model(repo="r", file="m.gguf", alias="m", parallel=1, args=["--temp", "1.0"])


def test_replica_command():
    cmd, env = common.llama_command("llama-server", MODEL, ("m.gguf", None), 8090, ["1"], 4, 65536, ["--metrics"])
    assert cmd == [
        "llama-server", "-m", "m.gguf", "--host", "127.0.0.1", "--port", "8090", "--alias", "m",
        "--no-webui", "--parallel", "4", "--kv-unified-per-slot", "65536", "--metrics", "--temp", "1.0",
    ]  # fmt: skip
    assert env["CUDA_VISIBLE_DEVICES"] == "1"


def test_split_command_with_draft_and_tensor_split():
    model = dataclasses.replace(MODEL, tensor_split="5,4")
    cmd, env = common.llama_command("llama-server", model, ("m.gguf", "d.gguf"), 8090, ["0", "1"], 1, 32768, [])
    assert cmd[cmd.index("--model-draft") + 1] == "d.gguf"
    assert cmd[cmd.index("--tensor-split") + 1] == "5,4" and "--split-mode" in cmd
    assert env["CUDA_VISIBLE_DEVICES"] == "0,1"


def test_gpu_stats_survive_a_missing_nvidia_smi(monkeypatch):
    monkeypatch.setenv("PATH", "")
    assert common.gpu_stats() == []


def test_publish_to_a_self_hosted_server_with_a_token(monkeypatch):
    sent = []
    monkeypatch.setattr(common.urllib.request, "urlopen", lambda req, timeout: sent.append(req))
    ntfy = Ntfy(server="https://ntfy.example.com/", token="tk_abc")
    common.publish(ntfy, "kis-x", Event(EventType.READY, t=3, session="s"), {"Title": "t"})
    (req,) = sent
    assert req.full_url == "https://ntfy.example.com/kis-x"
    assert req.get_header("Authorization") == "Bearer tk_abc" and req.get_header("Title") == "t"
    assert json.loads(req.data) == {"event": "ready", "t": 3, "session": "s"}
    common.publish(ntfy, "", "no topic: not sent")
    assert len(sent) == 1


def test_events_are_indented_unless_too_long_for_ntfy():
    short = common.event_message(Event(EventType.READY, t=3, session="s"))
    assert "\n  " in short and json.loads(short)["event"] == "ready"
    long = common.event_message(Event(EventType.ERROR, data={"log": "x" * common.NTFY_MESSAGE_BYTES}))
    assert "\n" not in long and json.loads(long)["log"].startswith("x")


def fake_proc(tmp_path, meminfo="MemTotal: 30720000 kB\nMemAvailable: 20480000 kB\n"):
    (tmp_path / "proc").mkdir()
    (tmp_path / "proc/meminfo").write_text(meminfo)
    return tmp_path / "proc"


def test_host_memory_reads_the_cgroup_without_the_page_cache(tmp_path, monkeypatch):
    cgroup = tmp_path / "cg"
    cgroup.mkdir()
    (cgroup / "memory.current").write_text(f"{10 << 30}\n")
    (cgroup / "memory.max").write_text(f"{16 << 30}\n")
    (cgroup / "memory.stat").write_text(f"anon 1\ninactive_file {2 << 30}\n")
    monkeypatch.setattr(common, "PROC", fake_proc(tmp_path))
    monkeypatch.setattr(common, "CGROUP", cgroup)
    assert common.host_memory() == (8 * 1024, 16 * 1024)


def test_host_memory_of_an_unlimited_cgroup_is_the_machines(tmp_path, monkeypatch):
    cgroup = tmp_path / "cg"
    cgroup.mkdir()
    (cgroup / "memory.current").write_text(f"{4 << 30}\n")
    (cgroup / "memory.max").write_text("max\n")
    monkeypatch.setattr(common, "PROC", fake_proc(tmp_path))
    monkeypatch.setattr(common, "CGROUP", cgroup)
    assert common.host_memory() == (4096, 30000)


def test_host_memory_reads_cgroup_v1(tmp_path, monkeypatch):
    cgroup = tmp_path / "cg/memory"
    cgroup.mkdir(parents=True)
    (cgroup / "memory.usage_in_bytes").write_text(f"{6 << 30}\n")
    (cgroup / "memory.limit_in_bytes").write_text(f"{12 << 30}\n")
    (cgroup / "memory.stat").write_text(f"total_inactive_file {1 << 30}\n")
    monkeypatch.setattr(common, "PROC", fake_proc(tmp_path))
    monkeypatch.setattr(common, "CGROUP", tmp_path / "cg")
    assert common.host_memory() == (5 * 1024, 12 * 1024)


def test_host_memory_falls_back_to_meminfo(tmp_path, monkeypatch):
    monkeypatch.setattr(common, "PROC", fake_proc(tmp_path))
    monkeypatch.setattr(common, "CGROUP", tmp_path / "none")
    assert common.host_memory() == (10000, 30000)
    monkeypatch.setattr(common, "PROC", tmp_path / "gone")
    assert common.host_memory() is None


def test_ram_stats_reports_the_anonymous_memory_of_each_llama_server(tmp_path, monkeypatch):
    proc = fake_proc(tmp_path)
    (proc / "42").mkdir()
    (proc / "42/status").write_text(
        "Name:\tllama-server\nVmRSS:\t 9000000 kB\nRssAnon:\t 3145728 kB\nRssFile:\t 1 kB\n"
    )
    monkeypatch.setattr(common, "PROC", proc)
    monkeypatch.setattr(common, "CGROUP", tmp_path / "none")
    stats = common.ram_stats({8090: 42, 8091: 43})  # 43 is gone
    assert stats == RamStats(used_mib=10000, total_mib=30000, processes={"8090": 3072})


def test_a_crash_event_fits_one_ntfy_message_and_names_the_exit(server):
    usage = UsageSummary(1165, 4845126, 336516, 3996715, 0.825, 605.5, 9.6, 0.568)
    stats = Stats(
        session="3ae4d5af", model="gemma", topology=Topology.REPLICAS, slots=24, parallel=12, ctx=32768,
        uptime_s=1752, idle_s=0, inflight=8, requests=1537, errors=18, usage={"gemma": usage, "embed": usage},
        gpus=[GpuStats(0, 10053, 15360, 78), GpuStats(1, 0, 15360, 0)], preset="gemma-4-e4b",
        resident=["embed"], cpu=CpuStats(4.0, 61, {"8090": 170}),
        ram=RamStats(9000, 29000, {"8090": 3000, "8091": 3000, "8070": 400}),
    )  # fmt: skip
    crash = {"crashed_port": 8091, "exit_code": -9, "exit": "killed by SIGKILL (...)", "log": 'slot "x"\n' * 300}
    extra = {**crash, "ram_before": dump(stats.ram), "ram_peak_mib": 28000}
    event = Event(EventType.STOPPED, 1787, "s", data={"reason": "llama-server exited", **extra, **dump(stats)})
    assert len(common.event_message(event).encode()) > common.NTFY_MESSAGE_BYTES
    event = common.fit_message(event)
    assert len(common.event_message(event).encode()) <= common.NTFY_MESSAGE_BYTES
    assert event.data["log"].endswith('slot "x"\n') and event.data["exit_code"] == -9
    note = server.notification(event)
    assert note and "llama-server exited (killed by SIGKILL" in note.message


def test_cpu_meter_reports_use_between_two_readings(tmp_path, monkeypatch):
    cgroup, proc = tmp_path / "cg", tmp_path / "proc"
    (proc / "42").mkdir(parents=True)
    cgroup.mkdir()
    (cgroup / "cpu.max").write_text("200000 100000\n")  # 2 cores
    clock = {"t": 100.0}
    monkeypatch.setattr(common, "PROC", proc)
    monkeypatch.setattr(common, "CGROUP", cgroup)
    monkeypatch.setattr(common.time, "monotonic", lambda: clock["t"])
    monkeypatch.setattr(common.os, "sysconf", lambda name: 100)  # clock ticks per second

    def set_cpu(container_s: float, process_ticks: int):
        (cgroup / "cpu.stat").write_text(f"usage_usec {int(container_s * 1e6)}\nuser_usec 1\n")
        # pid, (comm with a space), state, then ppid ... utime (field 14) and stime (15)
        (proc / "42/stat").write_text(f"42 (llama server) S {'0 ' * 10}{process_ticks} 0 0 0 0\n")

    meter = common.CpuMeter()
    set_cpu(10.0, 1000)
    assert meter.read({8090: 42}) is None  # one reading is not a rate
    clock["t"] += 20
    set_cpu(30.0, 1000 + 3000)  # 20 s later: 20 CPU s of 2 cores x 20 s = 50%; llama used 30 s = 150% of a core
    assert meter.read({8090: 42}) == CpuStats(cores=2.0, used_pct=50, processes={"8090": 150})
    clock["t"] += 1
    set_cpu(31.0, 4100)
    assert meter.read({8090: 42}) == CpuStats(cores=2.0, used_pct=50, processes={"8090": 150})  # too soon: repeated
