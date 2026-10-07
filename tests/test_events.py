"""Endpoint discovery from status events (kis/events.py)."""

import dataclasses
import json

from kis import events
from kis.schema import Event, EventType, Ntfy


def with_events(monkeypatch, *names_and_urls):
    msgs = [
        (str(i), Event.from_json({"event": name, **({"endpoint": url} if url else {})}))
        for i, (name, url) in enumerate(names_and_urls)
    ]
    monkeypatch.setattr(events, "fetch", lambda ntfy, topic, since: msgs)


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
    config.tunnel = dataclasses.replace(config.tunnel, public_url="https://llm.example.com")
    with_events(monkeypatch, ("ready", "https://ignored"))
    assert events.endpoint(config, "t") == "https://llm.example.com"


def session_events(*rows):
    """(session, event, endpoint, at, t) -> fetch() output."""
    return [
        (str(i), Event.from_json({"event": e, "session": s, "t": t, **({"endpoint": url} if url else {})}, at=at))
        for i, (s, e, url, at, t) in enumerate(rows)
    ]


def test_rollover_old_session_stopping_keeps_the_new_one(monkeypatch, config):
    rows = session_events(
        ("a", "ready", "https://a", 1000, 60),
        ("b", "starting", None, 40000, 0),
        ("b", "ready", "https://b", 40100, 90),
        ("a", "stopped", None, 40200, 39260),
    )
    monkeypatch.setattr(events, "fetch", lambda ntfy, topic, since: rows)
    assert events.current(config, "t") == events.Session("b", True, "https://b", 40010)


def test_while_the_new_session_starts_the_old_one_serves(monkeypatch, config):
    rows = session_events(("a", "ready", "https://a", 1000, 60), ("b", "starting", None, 40000, 0))
    monkeypatch.setattr(events, "fetch", lambda ntfy, topic, since: rows)
    assert events.endpoint(config, "t") == "https://a"
    rows.append(("9", Event(EventType.ERROR, t=50, session="b", at=40050)))
    assert events.endpoint(config, "t") == "https://a"  # a failed replacement doesn't hide the old session


def test_fetch_polls_the_configured_server_with_its_token(monkeypatch):
    seen = []

    class Response:
        def __enter__(self):
            return self

        def __exit__(self, *exc):
            pass

        def read(self):
            lines = [
                {"event": "open", "id": "0"},
                {"event": "message", "id": "1", "time": 5, "message": json.dumps({"event": "ready", "endpoint": "u"})},
                {"event": "message", "id": "2", "time": 6, "message": "a human note"},
            ]
            return "\n".join(map(json.dumps, lines)).encode()

    monkeypatch.setattr(events.retry, "urlopen", lambda req, timeout, what: seen.append(req) or Response())
    got = events.fetch(Ntfy("https://ntfy.example.com", "tk"), "kis-x", "12h")
    assert got == [("1", Event(EventType.READY, data={"endpoint": "u"}, at=5))]
    assert seen[0].full_url == "https://ntfy.example.com/kis-x/json?poll=1&since=12h"
    assert seen[0].get_header("Authorization") == "Bearer tk"
