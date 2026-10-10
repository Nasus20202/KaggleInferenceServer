from datetime import UTC, datetime

import pytest

from kis import status
from kis.schema import Stats, load


def test_format_quota():
    q = status.Quota(used_h=1.65, total_h=30.0, reset=datetime(2026, 10, 10, tzinfo=UTC))
    line = status.format_quota(q, now=datetime(2026, 10, 7, 5, 0, tzinfo=UTC))
    assert line.startswith("GPU quota: 28.4 h left of 30 h (1.6 h used), resets ")
    assert line.endswith("(in 2d 19h)")
    assert "unavailable" in status.format_quota(None)


def test_format_stats():
    stats = {
        "session": "s",
        "parallel": 2,
        "model": "m",
        "topology": "replicas",
        "slots": 4,
        "ctx": 131072,
        "requests": 3,
        "errors": 0,
        "inflight": 1,
        "uptime_s": 600,
        "idle_s": 0,
        "gpus": [{"gpu": 0, "used_mib": 13907, "total_mib": 15360, "util_pct": 87}],
        "cpu": {"cores": 4.0, "used_pct": 63, "processes": {"8090": 180}},
        "ram": {"used_mib": 12288, "total_mib": 30720, "processes": {"8090": 3072}},
        "usage": {
            "m": {
                "requests": 2,
                "tokens_in": 200,
                "tokens_out": 100,
                "tokens_cached": 160,
                "cache_rate": 0.8,
                "prefill_tps": 200.0,
                "decode_tps": 50.0,
                "draft_acceptance": None,
            }
        },
    }
    text = status.format_stats(load(Stats, stats))
    assert "4 slots x 131072 tokens" in text
    assert "GPU0: 13.6 / 15.0 GiB VRAM, 87% busy" in text
    assert "CPU: 63% of 4 cores (llama-server :8090 180%)" in text
    assert "RAM: 12.0 / 30.0 GiB (llama-server :8090 3.0 GiB)" in text
    assert "m: 2 completions, tokens in 200 (cached 160, 80%), out 100; prefill 200.0 tok/s" in text


def test_format_models_marks_the_loaded_preset():
    models = [
        {"id": "Qwen3.5-4B-Q4_K_M", "preset": "qwen35-4b", "status": "downloaded"},
        {"id": "gemma-4-12B-it-qat-UD-Q4_K_XL", "preset": "gemma-4-12b", "status": "loaded"},
        {"id": "embeddinggemma", "preset": "embed", "status": "loaded", "resident": True},
    ]
    assert status.format_models(models).splitlines() == [
        "  qwen35-4b    Qwen3.5-4B-Q4_K_M  downloaded",
        "* gemma-4-12b  gemma-4-12B-it-qat-UD-Q4_K_XL  loaded",
        "* embed        embeddinggemma  resident",
    ]


def test_format_models_shows_slots_and_context():
    models = [
        {"id": "A", "preset": "a", "status": "loaded", "context_length": 65536, "parallel": 4, "slots": 8},
        {"id": "B", "preset": "b", "status": "available", "context_length": 32768, "parallel": 9, "slots": None},
    ]
    assert status.format_models(models).splitlines() == [
        "* a  A  loaded  8 slots x 64K",
        "  b  B  available  9 per instance x 32K",
    ]


def test_logs_file_says_why_the_server_cannot_be_read(monkeypatch, capsys):
    import argparse

    from conftest import SECRETS

    from kis import cli, events
    from kis.client import AdminClient

    def unreachable(self, file, lines):
        raise OSError("certificate verify failed")

    monkeypatch.setattr(cli, "load_secrets", lambda: SECRETS)
    monkeypatch.setattr(events, "endpoint", lambda config, topic: "https://llm.example.com")
    monkeypatch.setattr(AdminClient, "logs", unreachable)
    with pytest.raises(SystemExit) as e:
        cli.cmd_logs(argparse.Namespace(file="server.log", lines=3, since="1h"), None)  # type: ignore[arg-type]
    assert "cannot read server.log from https://llm.example.com: certificate verify failed" in str(e.value)


def test_follow_reports_the_kernel_state_until_it_ends(monkeypatch, config, capsys):
    from kis import cli, kaggle

    states = iter([kaggle.KernelState.QUEUED, kaggle.KernelState.QUEUED, kaggle.KernelState.ERROR])
    monkeypatch.setattr(cli.events, "fetch", lambda *a: [])
    monkeypatch.setattr(cli.kaggle, "state", lambda *a: next(states))
    monkeypatch.setattr(cli, "KERNEL_CHECK_S", -1)
    monkeypatch.setattr(cli, "EVENT_POLL_S", 0)
    with pytest.raises(SystemExit, match="kernel ended: error"):
        cli.follow(config, "topic", "0", until_ready=True, slug="kis-server")
    assert capsys.readouterr().out.count("kernel: queued") == 1
