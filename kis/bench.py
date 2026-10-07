"""Aggregate generation throughput at increasing concurrency."""

import asyncio
import json
import statistics
import time
from dataclasses import dataclass

import aiohttp

TOPICS = [
    "a lighthouse keeper",
    "a Kubernetes outage",
    "a chess tournament",
    "a desert city",
    "an AI lab",
    "a jazz band",
]


@dataclass(frozen=True)
class Result:
    latency_s: float
    tokens: int  # completion tokens
    tok_per_s: float | None  # decode speed llama-server reports
    draft_acceptance: float | None


async def _request(session: aiohttp.ClientSession, url: str, model: str, max_tokens: int, i: int) -> Result:
    body = {
        "model": model,
        "max_tokens": max_tokens,
        "stream": False,
        "ignore_eos": True,  # llama.cpp extension: fixed-length outputs make levels comparable
        "messages": [
            {"role": "user", "content": f"Write a long, detailed story about {TOPICS[i % len(TOPICS)]} (#{i})."}
        ],
    }
    start = time.perf_counter()
    async with session.post(f"{url}/v1/chat/completions", json=body) as r:
        text = await r.text()
        if r.status != 200:
            raise SystemExit(f"HTTP {r.status}: {text[:300]}")
        data = json.loads(text)
    timings = data.get("timings", {})
    return Result(
        latency_s=time.perf_counter() - start,
        tokens=data["usage"]["completion_tokens"],
        tok_per_s=timings.get("predicted_per_second"),
        draft_acceptance=timings["draft_n_accepted"] / timings["draft_n"] if timings.get("draft_n") else None,
    )


async def _level(url: str, api_key: str, model: str, concurrency: int, requests: int, max_tokens: int) -> None:
    pending = list(range(requests))
    results: list[Result] = []
    headers = {"Authorization": f"Bearer {api_key}"}
    async with aiohttp.ClientSession(headers=headers, timeout=aiohttp.ClientTimeout(total=None)) as session:

        async def worker() -> None:
            while pending:
                results.append(await _request(session, url, model, max_tokens, pending.pop()))

        start = time.perf_counter()
        await asyncio.gather(*(worker() for _ in range(concurrency)))
        wall = time.perf_counter() - start

    tokens = sum(r.tokens for r in results)
    per_request = [r.tok_per_s for r in results if r.tok_per_s]
    acceptance = [r.draft_acceptance for r in results if r.draft_acceptance is not None]
    aggregate = tokens / wall
    tok_per_request = statistics.mean(per_request) if per_request else 0
    latency = statistics.median(r.latency_s for r in results)
    accepted = f"{statistics.mean(acceptance):.0%}" if acceptance else "-"
    print(
        f"{concurrency:>5} {requests:>6} {aggregate:>11.1f} {tok_per_request:>11.1f} {latency:>8.1f}s {accepted:>9}",
        flush=True,
    )


def run(url: str, api_key: str, model: str, levels: list[int], max_tokens: int, min_requests: int) -> None:
    print(f"{url}  model={model}  max_tokens={max_tokens}")
    print(f"{'conc':>5} {'reqs':>6} {'agg tok/s':>11} {'req tok/s':>11} {'p50 lat':>9} {'draft acc':>9}")
    for c in levels:
        asyncio.run(_level(url, api_key, model, c, max(min_requests, 2 * c), max_tokens))
