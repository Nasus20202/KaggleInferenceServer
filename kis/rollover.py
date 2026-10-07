"""Server sessions: start one, and replace a running one without downtime.

A Kaggle session lasts at most 12 hours. A rollover starts a replacement session
on the other kernel slug while the old one keeps serving, switches to it once it
is ready, lets the old one finish its in-flight requests, then shuts it down.
Sessions are told apart by the id in their events (and in /admin/stats), which
matters for a named tunnel, where both sessions share one URL.
"""

import dataclasses
import logging
import secrets
import time
from collections.abc import Callable
from http import HTTPStatus

from . import events, kaggle, status
from .client import AdminClient
from .schema import ENDED, Event, EventType, ServerConfig
from .settings import Config, load_state, save_state

SLUGS = (kaggle.SERVER, kaggle.SERVER + "-b")  # alternate, so a replacement never touches the running kernel
QUOTA_MARGIN_H = 2 / 60  # stop this long before the weekly GPU quota runs out
MIN_QUOTA_H = 0.25  # don't start a replacement session with less quota than this
GRACE_S = 30  # before draining the old session: longer than kis proxy's endpoint refresh
POLL_S = 5  # between checks of the event topic or the old session
log = logging.getLogger("kis.rollover")


class RolloverError(RuntimeError):
    pass


def capped_max_hours(max_hours: float, quota: status.Quota | None) -> float:
    """The session's max_hours, cut to end QUOTA_MARGIN_H before the GPU quota runs out."""
    if not quota:
        return max_hours
    return round(max(0.0, min(max_hours, quota.total_h - quota.used_h - QUOTA_MARGIN_H)), 3)


def launch(config: Config, server: ServerConfig, slug: str) -> str:
    """Push a server session with settings `server` on `slug`; return its session id."""
    session = secrets.token_hex(4)
    sent = dataclasses.replace(server, session=session, max_hours=capped_max_hours(server.max_hours, status.quota()))
    kaggle.push(config, slug, kaggle.SERVER_SCRIPT, {kaggle.CONFIG_PARAM: sent}, sources=[kaggle.build_source(config)])
    state = load_state()
    state.server, state.slug, state.session, state.model = server, slug, session, server.model.alias
    save_state(state)
    log.info("session %s started on %s (max %.2f h)", session, slug, sent.max_hours)
    return session


def wait_ready(
    config: Config,
    topic: str,
    session: str,
    since: str,
    timeout: float = 1800,
    on_event: Callable[[Event], None] | None = None,
) -> str:
    """Endpoint of `session` once it is ready; raise RolloverError if it fails or times out."""
    deadline = time.time() + timeout
    while time.time() < deadline:
        for msg_id, event in events.fetch(config.ntfy, topic, since):
            since = msg_id
            if event.session != session:
                continue
            if on_event:
                on_event(event)
            if event.type == EventType.READY and event.endpoint:
                return config.tunnel.public_url or event.endpoint
            if event.type in ENDED:
                raise RolloverError(f"session {session}: {event.type} {event.problem}")
        time.sleep(POLL_S)
    raise RolloverError(f"session {session} not ready after {timeout / 60:.0f} min")


def retire(
    url: str,
    api_key: str,
    session: str,
    drain_timeout: float = 900,
    poll: float = POLL_S,
    give_up: float = 1200,
    grace: float | None = None,
) -> None:
    """Shut `session` down once it has no requests in flight (or after drain_timeout).
    It first waits `grace`, so every `kis proxy` (they re-read the endpoint every 15 s)
    has switched to the new session. With a named tunnel the URL may reach the other
    session; then ask again, up to give_up."""
    time.sleep(GRACE_S if grace is None else grace)
    client = AdminClient(url, api_key)
    start = time.time()
    deadline = start + drain_timeout
    while True:
        try:
            stats = client.stats()
        except OSError:
            log.info("session %s is gone", session)
            return
        drained = stats and stats.session == session and (stats.inflight == 0 or time.time() > deadline)
        if stats and drained and client.shutdown(session) == HTTPStatus.OK:
            log.info("session %s shut down (%d requests in flight)", session, stats.inflight)
            return
        if time.time() > start + give_up:
            log.warning("could not reach session %s to shut it down; it stops at its max_hours", session)
            return
        time.sleep(poll)


def rollover(config: Config, topic: str) -> tuple[events.Session, str]:
    """Start a replacement for the current session and wait until it is ready.
    Return (old session, new endpoint); the caller switches over and retires the old one."""
    old = events.current(config, topic)
    if not old:
        raise RolloverError("no running session")
    state = load_state()
    if not state.server:
        raise RolloverError("no saved session settings; start the server with `kis up`")
    quota = status.quota()
    if quota and quota.left_h < MIN_QUOTA_H:
        raise RolloverError("not enough GPU quota left for a new session")
    slug = SLUGS[1] if state.slug == SLUGS[0] else SLUGS[0]
    since = str(int(time.time()))
    session = launch(config, state.server, slug)
    log.info("rollover: waiting for session %s to replace %s", session, old.session)
    return old, wait_ready(config, topic, session, since)
