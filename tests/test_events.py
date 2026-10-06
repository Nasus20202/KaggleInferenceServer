"""Endpoint discovery from status events (kis/events.py)."""

from kis import events


def with_events(monkeypatch, *names_and_urls):
    msgs = [
        (str(i), {"event": name, **({"endpoint": url} if url else {})}) for i, (name, url) in enumerate(names_and_urls)
    ]
    monkeypatch.setattr(events, "fetch", lambda topic, since: msgs)


def test_latest_ready_endpoint(monkeypatch, config):
    with_events(
        monkeypatch,
        ("ready", "https://old.trycloudflare.com"),
        ("tunnel_restart", None),
        ("ready", "https://new.trycloudflare.com"),
    )
    assert events.endpoint(config, "t") == "https://new.trycloudflare.com"


def test_no_endpoint_after_stop_or_error(monkeypatch, config):
    for last in ("stopped", "error"):
        with_events(monkeypatch, ("ready", "https://a.trycloudflare.com"), (last, None))
        assert events.endpoint(config, "t") is None


def test_no_events(monkeypatch, config):
    with_events(monkeypatch)
    assert events.endpoint(config, "t") is None


def test_named_tunnel_public_url_wins(monkeypatch, config):
    config["tunnel"]["public_url"] = "https://llm.example.com"
    with_events(monkeypatch, ("ready", "https://ignored"))
    assert events.endpoint(config, "t") == "https://llm.example.com"
