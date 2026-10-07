from datetime import UTC, datetime

from kis import status


def test_format_quota():
    q = {"used_h": 1.65, "total_h": 30.0, "reset": datetime(2026, 10, 10, tzinfo=UTC)}
    line = status.format_quota(q, now=datetime(2026, 10, 7, 5, 0, tzinfo=UTC))
    assert line.startswith("GPU quota: 28.4 h left of 30 h (1.6 h used), resets ")
    assert line.endswith("(in 2d 19h)")
    assert "unavailable" in status.format_quota(None)


def test_format_stats():
    stats = {
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
    text = status.format_stats(stats)
    assert "4 slots x 131072 tokens" in text
    assert "GPU0: 13.6 / 15.0 GiB VRAM, 87% busy" in text
    assert "m: 2 completions, tokens in 200 (cached 160, 80%), out 100; prefill 200.0 tok/s" in text
