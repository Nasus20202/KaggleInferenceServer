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


def session_events(*rows):
    """(session, event, endpoint, at, t) -> fetch() output."""
    return [
        (str(i), {"event": e, "session": s, "at": at, "t": t, **({"endpoint": url} if url else {})})
        for i, (s, e, url, at, t) in enumerate(rows)
    ]


def test_rollover_old_session_stopping_keeps_the_new_one(monkeypatch, config):
    rows = session_events(
        ("a", "ready", "https://a", 1000, 60),
        ("b", "starting", None, 40000, 0),
        ("b", "ready", "https://b", 40100, 90),
        ("a", "stopped", None, 40200, 39260),
    )
    monkeypatch.setattr(events, "fetch", lambda topic, since: rows)
    current = events.current(config, "t")
    assert (current["session"], current["endpoint"], current["started"]) == ("b", "https://b", 40010)


def test_while_the_new_session_starts_the_old_one_serves(monkeypatch, config):
    rows = session_events(("a", "ready", "https://a", 1000, 60), ("b", "starting", None, 40000, 0))
    monkeypatch.setattr(events, "fetch", lambda topic, since: rows)
    assert events.endpoint(config, "t") == "https://a"
    rows.append(("9", {"event": "error", "session": "b", "at": 40050, "t": 50}))
    assert events.endpoint(config, "t") == "https://a"  # a failed replacement doesn't hide the old session
