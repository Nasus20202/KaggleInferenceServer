"""Kaggle script: serve GGUF models with llama-server behind an
OpenAI-compatible, API-key protected balancer, exposed through a tunnel.

  replicas  model fits one T4 -> one llama-server per GPU (about 2x aggregate tok/s)
  split     larger model      -> one llama-server across both GPUs

One preset is loaded at a time. A request whose `model` names another preset of the
session (autoload), or POST /admin/load, swaps to it: the new preset is downloaded
while the loaded one keeps serving, then requests in flight finish, new ones wait,
and the instances are replaced. Requests without a known model name go to the loaded
preset. GET /v1/models lists every preset with its status.

Resident presets (server.resident), such as an embedding model, run on GPU 0 for the
whole session beside the swapped one: requests naming them go to their instance and
never swap.

Started by `kis up <model>`, which fills in CONFIG. Status events, the endpoint
URL, a periodic stats heartbeat and the final stats (with `stopped`) go to a
private ntfy topic; everything is
also logged with timestamps to the kernel log and /kaggle/working/server.log
(one line per request). Authenticated routes: GET /admin/stats (GPU memory and
load, slots, requests), GET /admin/logs?file=...&lines=N, POST /admin/shutdown.
The session stops after `idle_minutes` without requests, after `max_hours`, or
on /admin/shutdown.
"""

import asyncio
import enum
import json
import logging
import sys
import threading
import time
from dataclasses import dataclass

from backends import Backends, BackendsFailed, prefetch  # inlined by `kis`
from balancer import Balancer  # inlined by `kis`
from common import cpu_stats, fit_message, publish, ram_stats  # inlined by `kis`
from kis.schema import (  # inlined by `kis`
    SERVER_LOG,
    Event,
    EventType,
    ServerConfig,
    Stats,
    dump,
    load,
)
from state import STARTED, WORK, Runtime, log  # inlined by `kis`
from tunnels import TUNNELS  # inlined by `kis`

# <params> (replaced by `kis up`; to run by hand in the Kaggle UI, edit the rendered .kis/kis-server/server.py)
CONFIG = {
    "api_key": "change-me",
    "model": {
        "repo": "unsloth/Qwen3.5-4B-MTP-GGUF",
        "revision": "86835bf9949e4d14d6860f7910b1340ad4f271a9",
        "file": "Qwen3.5-4B-Q4_K_M.gguf",
        "alias": "Qwen3.5-4B-Q4_K_M",
        "parallel": 16,
    },
    "preset": "qwen35-4b",
    "models": {},
    "autoload": True,
    "prefetch": [],
    "resident": [],
    "ntfy": {"server": "https://ntfy.sh", "token": ""},
    "ntfy_topic": "",
    "session": "",
    "notify": {"topic": "", "token": "", "server": ""},
    "tunnel": {"kind": "cloudflared"},
    "idle_minutes": 10,
    "max_hours": 11.5,
    "replica_max_gb": 11.0,
    "ctx": 32768,
    "max_queue": None,
    "sticky_slack": 4,
    "args": ["--n-gpu-layers", "999", "--flash-attn", "auto", "--metrics"],
}
# </params>

SETTINGS = load(ServerConfig, CONFIG)  # checked and typed; see kis/schema.py for each field
MODEL = SETTINGS.model  # loaded first
PRESETS = SETTINGS.models or {SETTINGS.preset or MODEL.alias: MODEL}  # name -> preset the session can load
START = SETTINGS.preset if SETTINGS.preset in PRESETS else next(iter(PRESETS))
STATS_MINUTES = 10  # heartbeat with GPU and request stats on the ntfy topic


RUNTIME = Runtime()


def setup_logging() -> None:
    formatter = logging.Formatter("%(asctime)s %(levelname)s %(message)s")
    for handler in (logging.StreamHandler(sys.stdout), logging.FileHandler(WORK / SERVER_LOG)):
        handler.setFormatter(formatter)
        log.addHandler(handler)
    log.setLevel(logging.INFO)


class Priority(enum.StrEnum):
    """ntfy message priorities used here."""

    LOW = "low"
    DEFAULT = "default"
    HIGH = "high"


@dataclass(frozen=True)
class Notification:
    """A human-readable message for the phone app (ntfy tags are emoji names)."""

    title: str
    message: str
    priority: Priority = Priority.DEFAULT
    tags: str = ""

    def headers(self) -> dict[str, str]:
        return {"Title": self.title, "Priority": self.priority.value, "Tags": self.tags}


def notify(kind: EventType, **data: object) -> None:
    """Log a status event and publish it to the ntfy topic that `kis` follows."""
    event = fit_message(Event(kind, t=round(time.time() - STARTED), session=SETTINGS.session, data=data))
    log.info("event %s", json.dumps(event.to_json()))
    try:
        publish(SETTINGS.ntfy, SETTINGS.ntfy_topic, event)
    except OSError as e:
        log.warning("ntfy failed: %s", e)
    if (note := notification(event)) and SETTINGS.notify.topic:
        try:
            publish(SETTINGS.notify.via(SETTINGS.ntfy), SETTINGS.notify.topic, note.message, note.headers())
        except OSError as e:
            log.warning("notification failed: %s", e)


def notification(event: Event) -> Notification | None:
    """The phone notification for events worth one. Never includes the endpoint or keys:
    the notification topic may be public."""
    model = event.data.get("model") or RUNTIME.model or MODEL.alias
    if event.type == EventType.READY:
        layout = f"{RUNTIME.slots} slots x {RUNTIME.ctx} tokens"
        message = f"{layout}, ready after {event.t} s"
        return Notification(f"{model} ready", message, tags="white_check_mark")
    if event.type == EventType.STOPPED:
        how = event.data.get("exit")
        problem = f"{event.problem} ({how})" if how else event.problem
        return Notification(f"{model} stopped", f"{problem} after {round(event.t / 60)} min", tags="stop_sign")
    if event.type == EventType.ERROR:
        return Notification(f"{model} failed", event.problem[:300], Priority.HIGH, "warning")
    if event.type == EventType.RETRY:
        slots = event.data.get("parallel")
        return Notification(f"{model}: out of memory", f"retrying with {slots} slots", Priority.LOW, "hourglass")
    if event.type == EventType.MODEL_READY:
        layout = f"{event.data.get('slots')} slots x {event.data.get('ctx')} tokens"
        return Notification(
            f"{model} loaded", f"{layout}, after {event.data.get('seconds')} s", tags="arrows_counterclockwise"
        )
    if event.type == EventType.MODEL_FAILED:
        return Notification(f"{model} failed to load", event.problem[:300], Priority.HIGH, "warning")
    return None


# --------------------------------------------------------------------------- main


async def serve(backends: Backends) -> tuple[str, Stats, dict[str, object]]:
    """Run the balancer and tunnel until shutdown; return the stop reason, the final stats and,
    if a llama-server died, how (its exit, its log and the memory just before)."""
    balancer = Balancer(
        backends, SETTINGS.api_key, SETTINGS, RUNTIME, notify, PRESETS, START, residents=backends.residents
    )
    await balancer.start()
    tunnel = SETTINGS.tunnel

    def on_ready(url: str) -> None:
        notify(EventType.READY, endpoint=url, model=RUNTIME.model, preset=RUNTIME.preset, tunnel=tunnel.kind)

    threading.Thread(target=TUNNELS[tunnel.kind], args=(tunnel, on_ready, notify), daemon=True).start()
    if SETTINGS.prefetch:
        threading.Thread(target=prefetch, args=(SETTINGS.prefetch, PRESETS), daemon=True).start()

    reason, last_stats, extra = "shutdown requested", time.time(), {}
    ram, ram_peak_mib = None, 0  # host memory at the last check, and its peak this session
    while not balancer.stop.is_set():
        await asyncio.sleep(20)
        if time.time() - last_stats > STATS_MINUTES * 60:
            last_stats = time.time()
            notify(EventType.STATS, **dump(await balancer.stats()))
        if balancer.idle_seconds > SETTINGS.idle_minutes * 60:
            reason = f"idle for {SETTINGS.idle_minutes:g} min"
        elif time.time() - STARTED > SETTINGS.max_hours * 3600:
            reason = "max_hours reached"
        elif not balancer.loading and (crash := backends.crash()):
            reason = "llama-server exited"
            extra = {**crash, "ram_before": dump(ram), "ram_peak_mib": ram_peak_mib}
        else:
            ram = await asyncio.to_thread(ram_stats, backends.pids)
            await asyncio.to_thread(cpu_stats, backends.pids)  # keeps the CPU window short
            ram_peak_mib = max(ram_peak_mib, ram.used_mib if ram else 0)
            continue
        break
    stats = await balancer.stats()
    await balancer.close()
    return balancer.failure or reason, stats, extra


def main() -> None:
    setup_logging()
    notify(EventType.STARTING, model=MODEL.alias, preset=START)
    backends = Backends(SETTINGS, PRESETS, RUNTIME, notify)
    backends.fetch(START)
    notify(EventType.DOWNLOADED)
    try:
        backends.start_residents(SETTINGS.resident)
        backends.load(START)
    except BackendsFailed as e:
        notify(EventType.ERROR, error=str(e), log=e.log)
        backends.close()
        sys.exit(1)
    try:
        reason, stats, extra = asyncio.run(serve(backends))
    finally:
        backends.close()
    log = extra.pop("log", None)  # last: the longest field
    notify(EventType.STOPPED, reason=reason, **extra, **dump(stats), **({"log": log} if log else {}))


if __name__ == "__main__":
    try:
        main()
    except Exception as e:
        notify(EventType.ERROR, error=repr(e)[:2000])
        raise
