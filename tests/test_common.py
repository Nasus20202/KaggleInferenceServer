"""kaggle/common.py: the llama-server command shared by the server and calibration."""

import dataclasses
import json

import common
from kis.schema import Event, EventType, Model, Ntfy

MODEL = Model(repo="r", file="m.gguf", alias="m", parallel=1, args=["--temp", "1.0"])


def test_replica_command():
    cmd, env = common.llama_command("llama-server", MODEL, ("m.gguf", None), 8090, ["1"], 4, 65536, ["--metrics"])
    assert cmd == [
        "llama-server", "-m", "m.gguf", "--host", "127.0.0.1", "--port", "8090", "--alias", "m",
        "--parallel", "4", "--kv-unified-per-slot", "65536", "--metrics", "--temp", "1.0",
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
