"""Kaggle GPU quota and live stats of the running server, for `kis status` and `kis up`."""

from dataclasses import dataclass
from datetime import UTC, datetime
from typing import Any

from .schema import Stats


@dataclass(frozen=True)
class Quota:
    """Weekly Kaggle GPU quota."""

    used_h: float
    total_h: float
    reset: datetime | None = None  # UTC

    @property
    def left_h(self) -> float:
        return max(0.0, self.total_h - self.used_h)


def quota() -> Quota | None:
    """Weekly GPU quota: used and total hours and the reset time (UTC), or None if unavailable."""
    try:
        from kaggle import api

        q = api.quota_view()
    except Exception:  # offline, old CLI or auth problem: the quota is informational only
        return None
    gpu = q.gpu_quota
    if gpu is None:
        return None
    reset: datetime | None = q.quota_refresh_time
    return Quota(
        used_h=gpu.time_used.total_seconds() / 3600,
        total_h=gpu.total_time_allowed.total_seconds() / 3600,
        reset=reset.replace(tzinfo=UTC) if reset and reset.tzinfo is None else reset,
    )


def format_quota(q: Quota | None, now: datetime | None = None) -> str:
    if not q:
        return "GPU quota: unavailable (`kaggle quota`)"
    line = f"GPU quota: {q.left_h:.1f} h left of {q.total_h:.0f} h ({q.used_h:.1f} h used)"
    if q.reset:
        now = now or datetime.now(UTC)
        hours = max(0, int((q.reset - now).total_seconds() // 3600))
        local = q.reset.astimezone().strftime("%a %d %b %H:%M")
        line += f", resets {local} (in {hours // 24}d {hours % 24}h)"
    return line


def format_models(models: list[dict[str, Any]]) -> str:
    """One line per preset: name, alias, status, slots x context; the loaded one marked with *.
    Slots are per instance (`parallel`) until a preset runs."""
    width = max((len(m["preset"]) for m in models), default=0)
    lines = []
    for m in models:
        mark = "*" if m.get("status") == "loaded" else " "
        layout = ""
        if ctx := m.get("context_length"):
            slots = f"{m['slots']} slots" if m.get("slots") else f"{m.get('parallel')} per instance"
            layout = f"  {slots} x {ctx // 1024}K"
        status = "resident" if m.get("resident") else m.get("status", "")
        lines.append(f"{mark} {m['preset']:<{width}}  {m['id']}  {status}{layout}".rstrip())
    return "\n".join(lines)


def pct(x: float | None) -> str:
    return "-" if x is None else f"{x:.0%}"


def format_stats(s: Stats) -> str:
    lines = [
        f"model: {s.model}{f' ({s.preset})' if s.preset else ''}  {s.topology}, {s.slots} slots x {s.ctx} tokens",
        f"requests: {s.requests} ({s.errors} errors), {s.inflight} in flight, "
        f"up {s.uptime_s // 60} min, idle {s.idle_s // 60} min",
    ]
    if s.resident:
        lines.insert(1, f"resident: {', '.join(s.resident)}")
    for model, u in s.usage.items():
        lines.append(
            f"{model}: {u.requests} completions, tokens in {u.tokens_in} "
            f"(cached {u.tokens_cached}, {pct(u.cache_rate)}), out {u.tokens_out}; "
            f"prefill {u.prefill_tps or '-'} tok/s, decode {u.decode_tps or '-'} tok/s per request, "
            f"draft acceptance {pct(u.draft_acceptance)}"
        )
    for g in s.gpus:
        vram = f"{g.used_mib / 1024:.1f} / {g.total_mib / 1024:.1f} GiB VRAM"
        lines.append(f"GPU{g.gpu}: {vram}, {g.util_pct}% busy")
    return "\n".join(lines)
