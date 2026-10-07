"""Kaggle GPU quota and live stats of the running server, for `kis status` and `kis up`."""

import json
import urllib.request
from datetime import UTC, datetime

from . import retry


def quota() -> dict | None:
    """Weekly GPU quota: used and total hours and the reset time (UTC), or None if unavailable."""
    try:
        from kaggle import api

        q = api.quota_view()
    except Exception:  # offline, old CLI or auth problem: the quota is informational only
        return None
    gpu = q.gpu_quota
    reset = q.quota_refresh_time
    return {
        "used_h": gpu.time_used.total_seconds() / 3600,
        "total_h": gpu.total_time_allowed.total_seconds() / 3600,
        "reset": reset.replace(tzinfo=UTC) if reset and reset.tzinfo is None else reset,
    }


def format_quota(q: dict | None, now: datetime | None = None) -> str:
    if not q:
        return "GPU quota: unavailable (`kaggle quota`)"
    left = max(0.0, q["total_h"] - q["used_h"])
    line = f"GPU quota: {left:.1f} h left of {q['total_h']:.0f} h ({q['used_h']:.1f} h used)"
    if q["reset"]:
        now = now or datetime.now(UTC)
        hours = max(0, int((q["reset"] - now).total_seconds() // 3600))
        local = q["reset"].astimezone().strftime("%a %d %b %H:%M")
        line += f", resets {local} (in {hours // 24}d {hours % 24}h)"
    return line


def server_stats(url: str, api_key: str) -> dict:
    req = urllib.request.Request(f"{url}/admin/stats", headers={"Authorization": f"Bearer {api_key}"})
    with retry.urlopen(req, timeout=30, what="server stats") as r:
        return json.loads(r.read())


def pct(x: float | None) -> str:
    return "-" if x is None else f"{x:.0%}"


def format_stats(s: dict) -> str:
    lines = [
        f"model: {s['model']}  {s.get('topology', '?')}, {s.get('slots', '?')} slots x {s.get('ctx', '?')} tokens",
        f"requests: {s['requests']} ({s['errors']} errors), {s['inflight']} in flight, "
        f"up {s['uptime_s'] // 60} min, idle {s['idle_s'] // 60} min",
    ]
    for model, u in s.get("usage", {}).items():
        lines.append(
            f"{model}: {u['requests']} completions, tokens in {u['tokens_in']} "
            f"(cached {u['tokens_cached']}, {pct(u['cache_rate'])}), out {u['tokens_out']}; "
            f"prefill {u['prefill_tps'] or '-'} tok/s, decode {u['decode_tps'] or '-'} tok/s per request, "
            f"draft acceptance {pct(u['draft_acceptance'])}"
        )
    for g in s["gpus"]:
        vram = f"{g['used_mib'] / 1024:.1f} / {g['total_mib'] / 1024:.1f} GiB VRAM"
        lines.append(f"GPU{g['gpu']}: {vram}, {g.get('util_pct', '?')}% busy")
    return "\n".join(lines)
