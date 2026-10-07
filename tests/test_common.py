"""kaggle/common.py: the llama-server command shared by the server and calibration."""

import common

MODEL = {"alias": "m", "args": ["--temp", "1.0"]}


def test_replica_command():
    cmd, env = common.llama_command("llama-server", MODEL, ("m.gguf", None), 8090, ["1"], 4, 65536, ["--metrics"])
    assert cmd == [
        "llama-server", "-m", "m.gguf", "--host", "127.0.0.1", "--port", "8090", "--alias", "m",
        "--parallel", "4", "--kv-unified-per-slot", "65536", "--metrics", "--temp", "1.0",
    ]  # fmt: skip
    assert env["CUDA_VISIBLE_DEVICES"] == "1"


def test_split_command_with_draft_and_tensor_split():
    model = {**MODEL, "tensor_split": "5,4"}
    cmd, env = common.llama_command("llama-server", model, ("m.gguf", "d.gguf"), 8090, ["0", "1"], 1, 32768, [])
    assert cmd[cmd.index("--model-draft") + 1] == "d.gguf"
    assert cmd[cmd.index("--tensor-split") + 1] == "5,4" and "--split-mode" in cmd
    assert env["CUDA_VISIBLE_DEVICES"] == "0,1"


def test_gpu_stats_survive_a_missing_nvidia_smi(monkeypatch):
    monkeypatch.setenv("PATH", "")
    assert common.gpu_stats() == []
