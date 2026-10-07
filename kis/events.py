"""Status events published by the server kernel to a private ntfy topic."""

import json
import urllib.request
from dataclasses import dataclass, replace

from . import retry
from .schema import ENDED, Event, EventType, Ntfy
from .settings import Config

HISTORY = "12h"  # how far back to look for sessions: longer than a Kaggle session


@dataclass(frozen=True)
class Session:
    """One server session as its events describe it."""

    session: str
    alive: bool = True
    endpoint: str | None = None
    started: float = 0.0  # unix time
    preset: str = ""  # the loaded preset, as of the last ready or model_ready event


def fetch(ntfy: Ntfy, topic: str, since: str) -> list[tuple[str, Event]]:
    """(message id, event) pairs newer than `since` (a message id, unix time or duration like 12h)."""
    req = urllib.request.Request(f"{ntfy.url(topic)}/json?poll=1&since={since}", headers=ntfy.headers())
    try:
        with retry.urlopen(req, timeout=15, what="ntfy") as r:
            lines = r.read().decode().splitlines()
    except OSError as e:
        print(f"! ntfy: {e}")
        return []
    out = []
    for line in lines:
        msg = json.loads(line)
        if msg.get("event") != "message":  # ntfy's own open/keepalive messages
            continue
        try:
            out.append((msg["id"], Event.from_json(json.loads(msg["message"]), at=msg["time"])))
        except (ValueError, KeyError, TypeError, AttributeError):  # not an event sent by a kernel
            continue
    return out


def sessions(events: list[Event]) -> dict[str, Session]:
    """Server sessions in start order: endpoint, start time and whether they are still up.
    During a rollover two sessions run at once; each event names its session."""
    out: dict[str, Session] = {}
    for event in events:
        s = out.get(event.session) or Session(event.session)
        if event.type == EventType.READY:
            s = replace(s, endpoint=event.endpoint, started=event.at - event.t, preset=event.data.get("preset", ""))
        elif event.type == EventType.MODEL_READY:
            s = replace(s, preset=event.data.get("preset", s.preset))
        elif event.type in ENDED:
            s = replace(s, alive=False)
        out[event.session] = s
    return out


def current(config: Config, topic: str) -> Session | None:
    """The newest session that is ready and hasn't stopped, or None."""
    events = [e for _, e in fetch(config.ntfy, topic, HISTORY)]
    live = [s for s in sessions(events).values() if s.alive and s.endpoint]
    if not live:
        return None
    s = max(live, key=lambda s: s.started)
    return replace(s, endpoint=config.tunnel.public_url or s.endpoint)


def endpoint(config: Config, topic: str) -> str | None:
    """Base URL of the running server, or None when no session is up."""
    s = current(config, topic)
    return s.endpoint if s else None


def show(event: Event) -> None:
    data = dict(event.data)
    log = data.pop("log", None)
    print(f"[{event.t:>5}s] {event.type}: {json.dumps(data)}")
    if log:
        print("    " + str(log).replace("\n", "\n    "))
