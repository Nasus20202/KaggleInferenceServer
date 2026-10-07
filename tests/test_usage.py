"""Token and speed stats parsed from llama-server responses (kaggle/server.py)."""

import json

from aiohttp import web
from conftest import FakeBackends

from kis.schema import EventType, Model, Notify, Ntfy

AUTH = {"Authorization": "Bearer secret"}
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
    assert (u.model, u.tokens_in, u.tokens_out, u.cached) == ("m", 100, 50, 80)
    assert u.describe() == "in=100 cached=80 out=50 prefill=200tok/s decode=50.0tok/s draft=75%"


def test_parse_sse_stream_with_timings_only(server):
    events = [{"choices": [{"delta": {"content": "hi"}}]}, {"choices": [], "timings": TIMINGS}]
    body = b": keepalive\n\n" + b"".join(b"data: " + json.dumps(e).encode() + b"\n\n" for e in events)
    body += b"data: [DONE]\n\n"
    u = server.parse_usage(body, sse=True)
    assert (u.tokens_in, u.tokens_out, u.cached) == (100, 50, 80)


def test_no_usage(server):
    assert server.parse_usage(b'{"data": []}', sse=False) is None
    assert server.parse_usage(b"not json", sse=False) is None


def test_totals_per_model(server):
    usage = server.Usage()
    u = server.parse_usage(json.dumps({"usage": USAGE, "timings": TIMINGS}).encode(), sse=False)
    usage.add(u)
    usage.add(u)
    s = usage.summary()
    assert (s.requests, s.tokens_in, s.tokens_out, s.tokens_cached) == (2, 200, 100, 160)
    assert (s.cache_rate, s.prefill_tps, s.decode_tps, s.draft_acceptance) == (0.8, 200.0, 50.0, 0.75)


def test_notifications_are_readable_and_never_leak_the_endpoint(server, monkeypatch):
    sent = []

    def publish(ntfy, topic, payload, headers=None):
        sent.append((ntfy, topic, payload, headers))

    monkeypatch.setattr(server, "publish", publish)
    monkeypatch.setattr(server.SETTINGS, "ntfy", Ntfy("https://ntfy.example.com", "tk_control"))
    monkeypatch.setattr(server.SETTINGS, "ntfy_topic", "kis-private")
    monkeypatch.setattr(server.SETTINGS, "notify", Notify(topic="phone", token="tk", server="https://ntfy.sh"))
    monkeypatch.setattr(server.RUNTIME, "slots", 4)
    monkeypatch.setattr(server.RUNTIME, "ctx", 131072)
    server.notify(EventType.READY, endpoint="https://secret.trycloudflare.com")
    control, phone = sent
    assert control[:2] == (Ntfy("https://ntfy.example.com", "tk_control"), "kis-private")
    assert control[2].endpoint == "https://secret.trycloudflare.com"
    assert phone[:2] == (Ntfy("https://ntfy.sh", "tk"), "phone")
    assert "secret" not in phone[2] and "4 slots x 131072 tokens" in phone[2]
    assert phone[3]["Title"].endswith("ready")
    sent.clear()
    server.notify(EventType.STATS, requests=3)  # not worth a notification
    assert [topic for _, topic, _, _ in sent] == ["kis-private"]
    server.notify(EventType.ERROR, error="CUDA out of memory")
    assert sent[-1][3]["Priority"] == "high"


async def test_usage_of_responses_too_large_to_keep_whole(server, aiohttp_server, aiohttp_client):
    """Batches of embeddings: llama-server may write `usage` before or after the vectors."""
    vectors = [{"index": i, "embedding": [0.123456] * 768} for i in range(64)]  # ~600 KB of JSON
    orders = {
        "first": {"model": "E", "usage": {"prompt_tokens": 7}, "data": vectors},
        "last": {"data": vectors, "model": "E", "usage": {"prompt_tokens": 9}},
    }

    async def embeddings(req):
        return web.json_response(orders[(await req.json())["input"]])

    app = web.Application()
    app.router.add_post("/v1/embeddings", embeddings)
    embedder = await aiohttp_server(app)
    presets = {
        "a": Model(repo="r", file="a.gguf", alias="A", parallel=1),
        "e": Model(repo="r", file="e.gguf", alias="E", parallel=1),
    }
    backends = FakeBackends({"a": []}, "a")
    balancer = server.Balancer(backends, "secret", presets=presets, loaded="a", residents={"e": embedder.port})
    client = await aiohttp_client(balancer.app())
    for order in orders:
        resp = await client.post("/v1/embeddings", json={"model": "e", "input": order}, headers=AUTH)
        assert len((await resp.json())["data"]) == 64
    summary = balancer.usage["E"].summary()
    assert (summary.requests, summary.tokens_in) == (2, 16)
