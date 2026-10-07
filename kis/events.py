"""Status events published by the server kernel to a private ntfy.sh topic."""

import contextlib
import json

from . import retry


def fetch(topic: str, since: str) -> list[tuple[str, dict]]:
    """(message id, event) pairs newer than `since` (a message id, unix time or duration like 12h)."""
    try:
        with retry.urlopen(f"https://ntfy.sh/{topic}/json?poll=1&since={since}", timeout=15, what="ntfy") as r:
            lines = r.read().decode().splitlines()
    except OSError as e:
        print(f"! ntfy: {e}")
        return []
    out = []
    for line in lines:
        msg = json.loads(line)
        if msg.get("event") == "message":
            with contextlib.suppress(json.JSONDecodeError, TypeError):  # skip messages not sent by the server
                event = json.loads(msg["message"])
                event["at"] = msg["time"]  # unix time ntfy received it
                out.append((msg["id"], event))
    return out


def sessions(events: list[dict]) -> dict[str, dict]:
    """Server sessions in start order: endpoint, start time and whether they are still up.
    During a rollover two sessions run at once; each event names its session."""
    out: dict[str, dict] = {}
    for event in events:
        s = out.setdefault(event.get("session") or "", {"session": event.get("session") or "", "alive": True})
        if event["event"] == "ready":
            s.update(endpoint=event["endpoint"], started=event.get("at", 0) - event.get("t", 0))
        elif event["event"] in ("stopped", "error"):
            s["alive"] = False
    return out


def current(config: dict, topic: str) -> dict | None:
    """The newest session that is ready and hasn't stopped, or None."""
    live = [s for s in sessions([e for _, e in fetch(topic, "12h")]).values() if s["alive"] and "endpoint" in s]
    if not live:
        return None
    s = max(live, key=lambda s: s["started"])
    return {**s, "endpoint": config["tunnel"].get("public_url") or s["endpoint"]}


def endpoint(config: dict, topic: str) -> str | None:
    """Base URL of the running server, or None when no session is up."""
    s = current(config, topic)
    return s["endpoint"] if s else None


def show(event: dict):
    event = dict(event)
    event.pop("at", None)
    log = event.pop("log", None)
    print(f"[{event.pop('t', '?'):>5}s] {event.pop('event')}: {json.dumps(event)}")
    if log:
        print("    " + log.replace("\n", "\n    "))
