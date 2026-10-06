"""Aggregate generation throughput at increasing concurrency."""

import asyncio
import json
import statistics
import time

import aiohttp

TOPICS = [
    "a lighthouse keeper",
    "a Kubernetes outage",
    "a chess tournament",
    "a desert city",
    "an AI lab",
    "a jazz band",
]


async def _request(session: aiohttp.ClientSession, url: str, model: str, max_tokens: int, i: int):
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
    return time.perf_counter() - start, data["usage"]["completion_tokens"], data.get("timings", {})


async def _level(url: str, api_key: str, model: str, concurrency: int, requests: int, max_tokens: int):
    pending = list(range(requests))
    results = []
    headers = {"Authorization": f"Bearer {api_key}"}
    async with aiohttp.ClientSession(headers=headers, timeout=aiohttp.ClientTimeout(total=None)) as session:

        async def worker():
            while pending:
                results.append(await _request(session, url, model, max_tokens, pending.pop()))

        start = time.perf_counter()
        await asyncio.gather(*(worker() for _ in range(concurrency)))
        wall = time.perf_counter() - start

    tokens = sum(r[1] for r in results)
    per_request = [r[2]["predicted_per_second"] for r in results if r[2].get("predicted_per_second")]
    acceptance = [r[2]["draft_n_accepted"] / r[2]["draft_n"] for r in results if r[2].get("draft_n")]
    aggregate = tokens / wall
    tok_per_request = statistics.mean(per_request) if per_request else 0
    latency = statistics.median(r[0] for r in results)
    accepted = f"{statistics.mean(acceptance):.0%}" if acceptance else "-"
    print(
        f"{concurrency:>5} {requests:>6} {aggregate:>11.1f} {tok_per_request:>11.1f} {latency:>8.1f}s {accepted:>9}",
        flush=True,
    )


def run(url: str, api_key: str, model: str, levels: list[int], max_tokens: int, min_requests: int):
    print(f"{url}  model={model}  max_tokens={max_tokens}")
    print(f"{'conc':>5} {'reqs':>6} {'agg tok/s':>11} {'req tok/s':>11} {'p50 lat':>9} {'draft acc':>9}")
    for c in levels:
        asyncio.run(_level(url, api_key, model, c, max(min_requests, 2 * c), max_tokens))
