"""Status events published by the server kernel to a private ntfy.sh topic."""

import contextlib
import json
import urllib.request


def fetch(topic: str, since: str) -> list[tuple[str, dict]]:
    """(message id, event) pairs newer than `since` (a message id, unix time or duration like 12h)."""
    try:
        with urllib.request.urlopen(f"https://ntfy.sh/{topic}/json?poll=1&since={since}", timeout=15) as r:
            lines = r.read().decode().splitlines()
    except OSError as e:
        print(f"! ntfy: {e}")
        return []
    out = []
    for line in lines:
        msg = json.loads(line)
        if msg.get("event") == "message":
            with contextlib.suppress(json.JSONDecodeError):  # skip messages not sent by the server
                out.append((msg["id"], json.loads(msg["message"])))
    return out


def endpoint(config: dict, topic: str) -> str | None:
    """Base URL of the running server, or None when the last session has stopped."""
    for _, event in reversed(fetch(topic, "12h")):
        if event["event"] == "ready":
            return config["tunnel"].get("public_url") or event["endpoint"]
        if event["event"] in ("stopped", "error"):
            return None
    return None


def show(event: dict):
    event = dict(event)
    log = event.pop("log", None)
    print(f"[{event.pop('t', '?'):>5}s] {event.pop('event')}: {json.dumps(event)}")
    if log:
        print("    " + log.replace("\n", "\n    "))
