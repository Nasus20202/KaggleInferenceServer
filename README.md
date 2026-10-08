# Kaggle Inference Server

Run llama.cpp on Kaggle's free GPUs (2× T4, 30 h/week) and use it locally as an OpenAI-compatible API.

```
your app ─> kis proxy (127.0.0.1:8080) ─> tunnel ─> Kaggle: balancer ─> llama-server per GPU
```

Models that fit one T4 run as **one replica per GPU** behind a least-busy balancer, which gives about 2× throughput. Bigger models are split across both GPUs. The Kaggle session stops itself after 10 idle minutes; starting again takes about 1 minute for small models and about 4 for the 27B.

## Quick start

**1. One-time setup**

1. On kaggle.com → Settings: verify your phone number (needed for GPU and internet), then under **API** create a token and save it:
   ```bash
   mkdir -p ~/.kaggle && mv ~/Downloads/access_token ~/.kaggle/ && chmod 600 ~/.kaggle/access_token
   uv run kaggle config view   # shows your username and auth_method: ACCESS_TOKEN
   ```
   The older `~/.kaggle/kaggle.json` and the `KAGGLE_API_TOKEN` environment variable work too.
2. Install [uv](https://docs.astral.sh/uv/), then:
   ```bash
   uv sync
   cp examples/config.32k.toml config.toml   # set kaggle.username; see Models for the other profiles
   ```

**2. Build llama.cpp for T4** (once per llama.cpp version, ~25 min)

```bash
uv run kis build
```

**3. Start a model**

```bash
uv run kis up qwen35-4b
```

It prints progress (`downloaded` → `backends_ready` → `ready: https://….trycloudflare.com/v1`) and returns when the endpoint is ready. Measured from `kis up`: 50 s for `qwen35-4b` and 4 min for `qwen38-27b` (mostly the 18 GB download).

**4. Use it**

```bash
uv run kis proxy        # keep running: http://127.0.0.1:8080/v1, any API key
```

```bash
curl localhost:8080/v1/chat/completions -H 'Content-Type: application/json' \
  -d '{"messages":[{"role":"user","content":"Hello!"}]}'
```

Any OpenAI client works with `base_url="http://127.0.0.1:8080/v1"`.

By default, any client on your machine can use the proxy. Set `KIS_PROXY_API_KEY` to require a key: clients then send `Authorization: Bearer <key>` (or `x-api-key`), and other requests get 401. Do this when the proxy listens beyond localhost, as in Docker. The Kaggle server always has its own generated key; the proxy adds it.

A quick tunnel can lose its registration while `cloudflared` keeps running (clients then get 530, later no DNS). The server checks its own public URL every 30 s and restarts `cloudflared` after 3 failed checks, which announces a new URL that the proxy follows. Set `KIS_PROXY_GIVE_UP_MINUTES=10` to make the proxy exit once the server has been unreachable that long, so that CI jobs fail fast instead of retrying a dead URL.

The proxy retries transient failures with exponential backoff and jitter: a dropped connection, or a 502/503/504/524 while the tunnel reconnects. It re-reads the endpoint between retries and only retries before any bytes have reached the client. `kis` commands retry Kaggle, ntfy and server calls the same way.

**5. Measure and stop**

```bash
uv run kis bench --concurrency 1,8,16,32   # aggregate tok/s per concurrency level
uv run kis down                            # stops the session and saves GPU quota
```

## Commands

| Command | What it does |
| --- | --- |
| `kis up <model> [--parallel N] [--ctx N] [--topology split] [--no-spec]` | start a model (presets in `config.toml`); with a session up, swap to it |
| `kis use [<model>]` | swap the running session to another preset; without one, list the presets and their status |
| `kis proxy [--port P]` | local endpoint; follows the tunnel URL when it changes |
| `kis status` | GPU quota left and reset time, kernel states, VRAM and load of the running server |
| `kis bench` | throughput at each concurrency level |
| `kis logs [--file llama-8090.log]` | follow server events, or print a log file from the running server |
| `kis env` | print `OPENAI_*` exports for the direct endpoint |
| `kis down` | stop the server |
| `kis rollover` | replace the running session with a fresh one, without downtime |
| `kis build [--ref vX.Y.Z]` | compile llama.cpp on Kaggle |
| `kis calibrate [--write] [model...]` | measure slots, context and speed of each preset on Kaggle |

`parallel` is the number of slots per llama-server instance (2 replicas → 2× the slots) and `ctx` the context each slot is guaranteed. If they don't fit in VRAM, the server starts with fewer slots, never less context, and `backends_ready` shows what it got.

## Switching models

A session holds one model at a time, besides any resident presets, and can swap to any preset in `config.toml`. `kis up <model>` picks the one it starts with. To switch, name another preset in a request's `model`, by preset name or alias:

```bash
curl localhost:8080/v1/chat/completions -H 'Content-Type: application/json' \
  -d '{"model": "gemma-4-12b", "messages": [{"role": "user", "content": "Hello!"}]}'
```

or swap by hand with `kis use gemma-4-12b` (or `kis up gemma-4-12b` while a session runs). `kis use` alone lists the presets; `*` marks the loaded one:

```
$ kis use
  qwen35-4b    Qwen3.5-4B-Q4_K_M  downloaded  9 per instance x 32K
* gemma-4-12b  gemma-4-12B-it-qat-UD-Q4_K_XL  loaded  14 slots x 32K
  qwen38-27b   Qwen3.8-27B-UD-Q4_K_M  available  4 per instance x 32K
```

What a swap does:
1. **Download:** the new preset downloads while the loaded one keeps serving.
2. **Drain:** requests in flight finish; new ones, for any preset, wait.
3. **Load:** the llama-server instances are replaced, with the new preset's topology, slots and context.

Waiting requests get keepalive bytes, so a swap longer than Cloudflare's 100 s limit doesn't cut them. If the new preset fails to load, they get the error and the previous preset is loaded again. Requests without `model`, or with a name that isn't a preset (`gpt-4o`), go to the loaded preset and never swap, so existing clients keep working.

- **`/v1/models`** lists every preset by alias. Tools with a model picker show them all. Besides OpenAI's fields, each entry has:
  - `preset` and `status`: `loaded`, `loading`, `downloaded` or `available`.
  - `context_length`: the context each request gets (one slot's), under the name OpenRouter uses; not the model's trained context.
  - `parallel` (slots per instance), `slots` (in total) and `topology`. The loaded preset shows what it runs with after any out-of-memory retries. The others show their configured values, with `slots` (and `topology`, unless the preset sets it) `null` until they run.
- **`autoload = false`** in `[server]`: only `kis use` swaps; a request for another preset gets a 400 `model_not_loaded`.
- **`prefetch = ["gemma-4-12b"]`** in `[server]`: download those presets in the background after startup, so swapping to them only loads them.
- **`resident = ["embeddinggemma"]`** in `[server]`: run those presets on GPU 0 for the whole session, beside the swapped one, e.g. an embedding model next to the chat model for RAG. Requests that name them go to their instance and never swap; `/v1/models` lists them as `loaded` with `"resident": true`, and `kis use` as `resident`. They start first, so the swapped preset's out-of-memory retries leave room for them. `kis up` starts a preset that isn't resident. An embedding preset needs `--embeddings` in its `args`.
- **Overrides:** `--parallel`, `--ctx`, `--topology` and `--no-spec` apply to the preset `kis up` starts with; the others keep their calibrated values. Each preset remembers the slots that fit after an out-of-memory retry, so loading it again is quicker.
- **Rollover** starts the new session with the preset loaded at that point.

Clients that alternate between presets make the session swap back and forth, and each swap costs a model load. Keep such clients on one preset, make the other one resident, or set `autoload = false`. Downloaded files stay on the kernel's disk until the session ends.

## Monitoring and logs

```
$ kis status
GPU quota: 28.3 h left of 30 h (1.7 h used), resets Sat 10 Oct 02:00 (in 2d 18h)
kis-server: running
endpoint: https://….trycloudflare.com/v1
model: Qwen3.5-9B-Q4_K_M (qwen35-9b)  replicas, 4 slots x 114688 tokens
requests: 12 (0 errors), 1 in flight, up 25 min, idle 0 min
GPU0: 13.6 / 15.0 GiB VRAM, 87% busy
GPU1: 13.6 / 15.0 GiB VRAM, 64% busy
CPU: 58% of 4 cores (llama-server :8090 171%, llama-server :8091 160%)
RAM: 11.2 / 29.0 GiB (llama-server :8090 3.4 GiB, llama-server :8091 3.3 GiB)
```

`kis up` also prints the quota before it starts a session. `kis status` also shows the token totals per model (see below).

### Phone notifications

Set `[notify] topic` in `config.toml` and subscribe to that topic in the [ntfy](https://ntfy.sh) app. You get:
- **Ready:** the model, slots and startup time.
- **Stopped:** the reason and how long the session ran.
- **Failed:** the error.
- **Out of memory:** each retry with fewer slots.
- **Model swaps:** the new model once it is loaded, or why it failed to load.

These messages never contain the endpoint or keys, so a guessable topic name is fine for them. `kis` uses its own private, random topic (in `.kis/secrets.json`) to find the server; never point that one at a public name, because anyone who can post to it could redirect `kis` to their own URL and receive your API key. `[notify] token` (or `KIS_NOTIFY_TOKEN`) is for access-protected topics.

### Self-hosted ntfy

`kis` uses [ntfy.sh](https://ntfy.sh) by default. To use your own ntfy server instead, set `[ntfy]`:

```toml
[ntfy]
server = "https://ntfy.example.com"
token = ""   # or KIS_NTFY_TOKEN; for servers that require login (tk_... access token)
```

The Kaggle kernel publishes there and `kis` reads from there, so the server must be reachable from both. Notifications go to the same server with the same token, unless `[notify] server` (and `[notify] token`) name another one. With `auth-default-access: deny-all`, give the token's user read-write access to the control topic (`kis-*`) and the `kis-*-calibrate` topic.

### Stats API

`GET /admin/stats` on the Kaggle server returns the same data as JSON. It needs the server's API key (`kis env` prints it). Through `kis proxy`, `curl localhost:8080/admin/stats` works without it (with `KIS_PROXY_API_KEY` if you set one):

```json
{
  "session": "3f9a1c2e", "preset": "qwen35-4b", "model": "Qwen3.5-4B-Q4_K_M", "topology": "replicas",
  "slots": 4, "parallel": 2, "ctx": 131072, "resident": [],
  "uptime_s": 1500, "idle_s": 3, "requests": 12, "inflight": 1, "errors": 0,
  "usage": {"Qwen3.5-4B-Q4_K_M": {
    "requests": 11, "tokens_in": 52310, "tokens_out": 8120, "tokens_cached": 41870, "cache_rate": 0.8,
    "prefill_tps": 1450.2, "decode_tps": 68.4, "draft_acceptance": 0.71}},
  "gpus": [{"gpu": 0, "used_mib": 12711, "total_mib": 15360, "util_pct": 87}, {"gpu": 1, "...": "..."}],
  "cpu": {"cores": 4.0, "used_pct": 58, "processes": {"8090": 171, "8091": 160}},
  "ram": {"used_mib": 11468, "total_mib": 29696, "processes": {"8090": 3481, "8091": 3380}}
}
```

`cpu` is the container's CPU use since the previous reading (every 20 s), as a share of all its cores, and each llama-server's as a share of one core. `ram` is the host memory of the Kaggle container, without the page cache: `processes` has each llama-server's anonymous memory (by port, in MiB), where its prompt cache lives, not the memory-mapped model. The server samples it every 20 s.

In `usage`:
- **Tokens:** `tokens_in` counts every prompt token, and `tokens_cached` the ones served from llama.cpp's prompt cache; `cache_rate` = cached / in.
- **Speeds:** `prefill_tps` and `decode_tps` are per-request speeds, weighted by time.
- **Draft acceptance:** the share of MTP draft tokens accepted.

`GET /admin/logs?file=server.log&lines=200` returns a log file (`kis logs --file`). `POST /admin/load?model=<preset>` starts a swap and answers 202 at once (`kis use` then follows the events).

The places to look:
- **Events:** start, ready, errors, a stats heartbeat every 10 min, and the final stats with `stopped`. `kis logs` follows them. If a llama-server dies, `stopped` also has `exit` and `exit_code` (a kill by the kernel's OOM killer shows as SIGKILL), `crashed_port`, `log` (the end of its output) and `ram_before` / `ram_peak_mib` (host memory at the last 20 s check and its peak).
- **Server log:** timestamped, one line per request: backend, status, duration, tokens in/cached/out, prefill and decode speed, draft acceptance. `kis logs --file server.log` prints it; it's also in the Kaggle kernel log.
- **llama-server output:** `kis logs --file llama-8090.log` (`llama-8091.log` for the second replica).
- **Local:** `kis proxy` logs each request and endpoint changes. `-v` or `KIS_LOG_LEVEL=DEBUG` adds more detail.

## Long runs, quota and overload

A Kaggle session lasts at most 12 hours. While `kis proxy` runs, it replaces a session after `rollover_hours` (11 by default):
1. **Start:** it launches a replacement on a second kernel (`kis-server-b`, alternating with `kis-server`) while the old session keeps serving.
2. **Switch:** once the replacement is ready, new requests go to it.
3. **Drain:** the old session finishes its in-flight requests and is shut down.

Clients see no errors, but the new session starts with an empty prompt cache. For about 5 minutes both sessions run, which costs double GPU quota. `kis rollover` does the same by hand. Each session has an id in its events, so `kis` can tell the two sessions apart while both run.

Sessions also stop 2 minutes before the weekly GPU quota runs out, with a clean `stopped` event. A rollover isn't started with less than 15 minutes of quota left.

`max_queue` (unset by default) caps the requests waiting beyond the slots. When the cap is reached, the server answers 429 with `Retry-After: 5`, and `kis proxy` retries with backoff.

## Models

`examples/` has one config per context profile. Every profile holds the same presets and fills each slot's context first, then adds as many slots as fit:

| file | context per slot |
| --- | --- |
| `config.32k.toml` | 32K |
| `config.64k.toml` | 64K |
| `config.96k.toml` | 96K |
| `config.128k.toml` | 128K |
| `config.max.toml` | the largest that fits, up to the model's trained context |

A model that doesn't fit one T4 at a profile's context is split across both GPUs. A model that doesn't fit even split is left out of that file. The KV cache stays in f16, so quality isn't reduced.

Measured with `kis calibrate` on Kaggle's 2× T4 (llama.cpp v0.6.0). Each cell shows slots × context per slot, then decode speed: one request alone / all slots busy (total tok/s). Speeds come from short prompts with 256 generated tokens and MTP; long contexts decode slower.

<!-- calibration -->
| model | 32k | 64k | 96k | 128k | max |
| --- | --- | --- | --- | --- | --- |
| `gemma-4-e2b` | 64 x 32K<br>97.9 / 696.2 tok/s | 64 x 64K<br>109.5 / 670.0 tok/s | 44 x 96K<br>109.0 / 690.8 tok/s | 32 x 128K<br>105.4 / 662.2 tok/s | 32 x 128K<br>102.7 / 674.4 tok/s |
| `gemma-4-e4b` | 42 x 32K<br>65.8 / 587.8 tok/s | 22 x 64K<br>68.3 / 498.0 tok/s | 14 x 96K<br>62.3 / 429.6 tok/s | 10 x 128K<br>77.4 / 386.6 tok/s | 10 x 128K<br>67.0 / 380.8 tok/s |
| `qwen35-4b` | 18 x 32K<br>44.1 / 203.8 tok/s | 8 x 64K<br>49.6 / 184.2 tok/s | 6 x 96K<br>43.1 / 163.4 tok/s | 4 x 128K<br>47.5 / 124.4 tok/s | 2 x 256K<br>42.4 / 89.4 tok/s |
| `qwen35-9b` | 14 x 32K<br>22.6 / 153.4 tok/s | 6 x 64K<br>29.8 / 125.8 tok/s | 4 x 96K<br>27.2 / 71.4 tok/s | 2 x 128K<br>30.2 / 56.0 tok/s | 1 x 256K split<br>34.7 / 37.1 tok/s |
| `gemma-4-e4b-q8` | 32 x 32K<br>37.4 / 473.6 tok/s | 16 x 64K<br>46.9 / 392.8 tok/s | 10 x 96K<br>44.7 / 301.8 tok/s | 8 x 128K<br>46.6 / 258.4 tok/s | 8 x 128K<br>45.9 / 255.8 tok/s |
| `qwen35-4b-q8` | 14 x 32K<br>40.3 / 196.2 tok/s | 8 x 64K<br>47.3 / 175.4 tok/s | 4 x 96K<br>43.5 / 120.8 tok/s | 4 x 128K<br>45.9 / 127.0 tok/s | 2 x 256K<br>40.5 / 83.0 tok/s |
| `qwen35-9b-q8` | 8 x 32K<br>23.1 / 119.0 tok/s | 4 x 64K<br>27.3 / 75.8 tok/s | 2 x 96K<br>25.3 / 50.0 tok/s | 2 x 128K<br>27.5 / 57.4 tok/s | 1 x 256K split<br>32.9 / 30.4 tok/s |
| `gemma-4-12b` | 14 x 32K<br>29.6 / 211.8 tok/s | 10 x 64K<br>33.2 / 206.4 tok/s | 6 x 96K<br>30.1 / 142.2 tok/s | 4 x 128K<br>30.9 / 90.8 tok/s | 2 x 256K<br>25.8 / 51.0 tok/s |
| `gemma-4-26b-a4b` | 14 x 32K split<br>68.6 / 121.7 tok/s | 8 x 64K split<br>69.3 / 116.1 tok/s | 5 x 96K split<br>66.3 / 108.0 tok/s | 4 x 128K split<br>69.4 / 108.2 tok/s | 2 x 256K split<br>62.6 / 77.2 tok/s |
| `qwen38-27b` | 4 x 32K split<br>12.0 / 29.9 tok/s | 2 x 64K split<br>13.1 / 15.2 tok/s | 1 x 96K split<br>13.1 / 11.6 tok/s | 1 x 128K split<br>12.7 / 12.1 tok/s | 1 x 176K split<br>14.1 / 12.4 tok/s |
<!-- /calibration -->

Calibration stops at 32 slots per llama-server instance, so `gemma-4-e2b` at 32K and 64K would fit more slots. "split" means one instance across both GPUs; otherwise each GPU runs a replica. With one slot alone, decode speed is set mostly by model size, not context size.

Each preset pins a revision and sha256. `kis up <model> --parallel N --ctx N` overrides a preset for one run. After a llama.cpp update, or to add presets, run `kis calibrate --write [model...]`; it regenerates `examples/` and this table.

## Tunnels

Set `[tunnel] kind` in `config.toml`:

- **`cloudflared`** (default): no account needed, random URL per session. Streaming (SSE) works in practice, although Cloudflare doesn't document it for quick tunnels.
- **`cloudflared` + `token` + `public_url`**: a named tunnel with a stable hostname on your Cloudflare domain. Point it to `http://localhost:8080`.
- **`tailscale` + `authkey`**: private tailnet hostname (use a reusable, ephemeral key).

Cloudflare cuts a request whose response hasn't started within 100 s, such as a long prompt without streaming. If a reply hasn't started after 30 s, the server sends a 200 status and keeps the connection alive with whitespace (JSON) or comments (SSE). An error that comes after that point arrives with status 200.

## Docker

```bash
cp examples/config.32k.toml config.toml && mkdir -p .kis
docker compose run --rm kis up qwen35-4b
docker compose up -d     # proxy on 127.0.0.1:8080
```

Image: `ghcr.io/nasus20202/kaggleinferenceserver`.

Every setting can also come from the environment, as `KIS_` and its path in upper case: `KIS_SERVER_ROLLOVER_HOURS=0` for `[server] rollover_hours`, `KIS_API_KEY` and `KIS_NTFY_TOPIC` for `.kis/secrets.json`. Strings are taken as they are, other values are read as TOML (`KIS_SERVER_ARGS='["--metrics"]'`); presets (`[models.*]`) only come from the file. In CI, pass the secrets that way instead of mounting `.kis`, and the config file with `KIS_CONFIG`. `kis proxy` needs no config file: it only follows the ntfy topic to the current session, so it runs as a service container with just the two secrets.

## Development

```bash
uv run pytest && uv run ruff format . && uv run ruff check . && uv run pyright
```

The code is fully typed and pyright checks it in standard mode. Settings, events, stats and calibration results are dataclasses in `kis/schema.py`, shared by `kis` and the Kaggle scripts. `kis` sends them to a kernel as plain data, and the kernel reads them back with `schema.load`, which checks every key and type. A typo in `config.toml` therefore fails at once with its path, e.g. `config.server: unknown key(s) idle_minuts`. Fixed sets of values are enums: `Topology`, `TunnelKind`, `EventType`, `Route`, `ModelStatus`.

CI runs these checks, then builds the image and publishes it to GHCR from `main`. Renovate automerges minor and patch updates and checks weekly for new llama.cpp releases. CI doesn't build llama.cpp, so after a llama.cpp update, run `kis build` again.

`kaggle/` holds the scripts that run on Kaggle; `kis/` is the local CLI. Kaggle runs one file per kernel, so `kis` pastes the modules a script imports into it when it pushes (`INLINED` in `kis/kaggle.py`). `server.py` holds the settings, notifications and startup; `state.py` (paths, ports, logger, `Runtime`), `backends.py` (llama-server instances), `balancer.py` (the API-key protected proxy, swaps and `/admin`) and `tunnels.py` (cloudflared, tailscale) are pasted into it, as are `common.py` and `kis/schema.py`. They share one namespace there: import them as `from module import a, b`, keep top-level names unique (`tests/test_render.py` checks), and pass the settings, runtime and `notify` as arguments. Running a public endpoint from Kaggle is a grey area under its terms, so keep the kernel private (the default) and for personal use.
