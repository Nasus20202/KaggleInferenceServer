"""Server sessions: start one, and replace a running one without downtime.

A Kaggle session lasts at most 12 hours. A rollover starts a replacement session
on the other kernel slug while the old one keeps serving, switches to it once it
is ready, lets the old one finish its in-flight requests, then shuts it down.
Sessions are told apart by the id in their events (and in /admin/stats), which
matters for a named tunnel, where both sessions share one URL.
"""

import json
import logging
import secrets
import time
import urllib.error
import urllib.request

from . import events, kaggle, retry, status
from .settings import load_state, save_state

SLUGS = (kaggle.SERVER, kaggle.SERVER + "-b")  # alternate, so a replacement never touches the running kernel
QUOTA_MARGIN_H = 2 / 60  # stop this long before the weekly GPU quota runs out
MIN_QUOTA_H = 0.25  # don't start a replacement session with less quota than this
GRACE_S = 30  # before draining the old session: longer than kis proxy's endpoint refresh
log = logging.getLogger("kis.rollover")


class RolloverError(RuntimeError):
    pass


def capped_max_hours(max_hours: float, quota: dict | None) -> float:
    """The session's max_hours, cut to end QUOTA_MARGIN_H before the GPU quota runs out."""
    if not quota:
        return max_hours
    return round(max(0.0, min(max_hours, quota["total_h"] - quota["used_h"] - QUOTA_MARGIN_H)), 3)


def launch(config: dict, params: dict, slug: str) -> str:
    """Push a server session with `params` on `slug`; return its session id."""
    session = secrets.token_hex(4)
    server: dict = {**params["CONFIG"], "session": session}
    server["max_hours"] = capped_max_hours(server["max_hours"], status.quota())
    kaggle.push(config, slug, "server.py", {"CONFIG": server}, sources=[kaggle.kernel_id(config, kaggle.BUILD)])
    save_state(server_params=params, slug=slug, session=session, model=server["model"]["alias"])
    log.info("session %s started on %s (max %.2f h)", session, slug, server["max_hours"])
    return session


def wait_ready(config: dict, topic: str, session: str, since: str, timeout: float = 1800, on_event=None) -> str:
    """Endpoint of `session` once it is ready; raise RolloverError if it fails or times out."""
    deadline = time.time() + timeout
    while time.time() < deadline:
        for msg_id, event in events.fetch(topic, since):
            since = msg_id
            if event.get("session") != session:
                continue
            if on_event:
                on_event(event)
            if event["event"] == "ready":
                return config["tunnel"].get("public_url") or event["endpoint"]
            if event["event"] in ("error", "stopped"):
                raise RolloverError(f"session {session}: {event['event']} {event.get('error') or event.get('reason')}")
        time.sleep(5)
    raise RolloverError(f"session {session} not ready after {timeout / 60:.0f} min")


def admin(url: str, api_key: str, path: str, method: str = "GET") -> tuple[int, dict]:
    req = urllib.request.Request(f"{url}{path}", method=method, headers={"Authorization": f"Bearer {api_key}"})
    try:
        with retry.urlopen(req, timeout=30, what=path) as r:
            return r.status, json.loads(r.read())
    except urllib.error.HTTPError as e:
        try:
            return e.code, json.loads(e.read() or b"{}")
        except ValueError:
            return e.code, {}


def retire(
    url: str,
    api_key: str,
    session: str,
    drain_timeout: float = 900,
    poll: float = 5,
    give_up: float = 1200,
    grace: float | None = None,
):
    """Shut `session` down once it has no requests in flight (or after drain_timeout).
    It first waits `grace`, so every `kis proxy` (they re-read the endpoint every 15 s)
    has switched to the new session. With a named tunnel the URL may reach the other
    session; then ask again, up to give_up."""
    time.sleep(GRACE_S if grace is None else grace)
    start = time.time()
    deadline = start + drain_timeout
    while True:
        try:
            code, stats = admin(url, api_key, "/admin/stats")
        except OSError:
            log.info("session %s is gone", session)
            return
        if code == 200 and stats.get("session") == session and (stats["inflight"] == 0 or time.time() > deadline):
            code, _ = admin(url, api_key, f"/admin/shutdown?session={session}", method="POST")
            if code == 200:
                log.info("session %s shut down (%d requests in flight)", session, stats["inflight"])
                return
        if time.time() > start + give_up:
            log.warning("could not reach session %s to shut it down; it stops at its max_hours", session)
            return
        time.sleep(poll)


def rollover(config: dict, topic: str) -> tuple[dict, str]:
    """Start a replacement for the current session and wait until it is ready.
    Return (old session, new endpoint); the caller switches over and retires the old one."""
    old = events.current(config, topic)
    if not old:
        raise RolloverError("no running session")
    state = load_state()
    if "server_params" not in state:
        raise RolloverError("no saved session settings; start the server with `kis up`")
    quota = status.quota()
    if quota and quota["total_h"] - quota["used_h"] < MIN_QUOTA_H:
        raise RolloverError("not enough GPU quota left for a new session")
    slug = SLUGS[1] if state.get("slug") == SLUGS[0] else SLUGS[0]
    since = str(int(time.time()))
    session = launch(config, state["server_params"], slug)
    log.info("rollover: waiting for session %s to replace %s", session, old["session"])
    return old, wait_ready(config, topic, session, since)
