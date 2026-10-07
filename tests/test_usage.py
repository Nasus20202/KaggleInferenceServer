"""Token and speed stats parsed from llama-server responses (kaggle/server.py)."""

import json

USAGE = {"prompt_tokens": 100, "completion_tokens": 50, "prompt_tokens_details": {"cached_tokens": 80}}
TIMINGS = {
    "prompt_n": 20,
    "prompt_ms": 100.0,
    "predicted_n": 50,
    "predicted_ms": 1000.0,
    "draft_n": 40,
    "draft_n_accepted": 30,
    "cache_n": 80,
}


def test_parse_json_body(server):
    body = b"   " + json.dumps({"model": "m", "usage": USAGE, "timings": TIMINGS}).encode()  # keepalive spaces
    u = server.parse_usage(body, sse=False)
    assert (u["model"], u["in"], u["out"], u["cached"]) == ("m", 100, 50, 80)
    assert server.describe(u) == "in=100 cached=80 out=50 prefill=200tok/s decode=50.0tok/s draft=75%"


def test_parse_sse_stream_with_timings_only(server):
    events = [{"choices": [{"delta": {"content": "hi"}}]}, {"choices": [], "timings": TIMINGS}]
    body = b": keepalive\n\n" + b"".join(b"data: " + json.dumps(e).encode() + b"\n\n" for e in events)
    body += b"data: [DONE]\n\n"
    u = server.parse_usage(body, sse=True)
    assert (u["in"], u["out"], u["cached"]) == (100, 50, 80)


def test_no_usage(server):
    assert server.parse_usage(b'{"data": []}', sse=False) is None
    assert server.parse_usage(b"not json", sse=False) is None


def test_totals_per_model(server):
    usage = server.Usage()
    u = server.parse_usage(json.dumps({"usage": USAGE, "timings": TIMINGS}).encode(), sse=False)
    usage.add(u)
    usage.add(u)
    s = usage.summary()
    assert (s["requests"], s["tokens_in"], s["tokens_out"], s["tokens_cached"]) == (2, 200, 100, 160)
    assert (s["cache_rate"], s["prefill_tps"], s["decode_tps"], s["draft_acceptance"]) == (0.8, 200.0, 50.0, 0.75)
