# Kaggle Inference Server

Run llama.cpp on Kaggle's free GPUs (2× T4, 30 h/week) and use it locally as an OpenAI-compatible API.

```
your app ─> kis proxy (127.0.0.1:8080) ─> tunnel ─> Kaggle: balancer ─> llama-server per GPU
```

Models that fit one T4 run as **one replica per GPU** behind a least-busy balancer, which gives about 2× throughput. Bigger models are split across both GPUs. The Kaggle session stops itself after 30 idle minutes.

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
   cp config.example.toml config.toml   # set kaggle.username; config.throughput.example.toml for many slots
   ```

**2. Build llama.cpp for T4** (once per llama.cpp version, ~25 min)

```bash
uv run kis build
```

**3. Start a model**

```bash
uv run kis up qwen35-4b
```

It prints progress (`downloaded` → `backends_ready` → `ready: https://….trycloudflare.com/v1`) and returns when the endpoint is ready, usually in 2–5 min.

**4. Use it**

```bash
uv run kis proxy        # keep running: http://127.0.0.1:8080/v1, any API key
```

```bash
curl localhost:8080/v1/chat/completions -H 'Content-Type: application/json' \
  -d '{"messages":[{"role":"user","content":"Hello!"}]}'
```

Any OpenAI client works with `base_url="http://127.0.0.1:8080/v1"`.

**5. Measure and stop**

```bash
uv run kis bench --concurrency 1,8,16,32   # aggregate tok/s per concurrency level
uv run kis down                            # stops the session and saves GPU quota
```

## Commands

| Command | What it does |
| --- | --- |
| `kis up <model> [--parallel N] [--ctx N] [--topology split] [--no-spec]` | start a model (presets in `config.toml`) |
| `kis proxy [--port P]` | local endpoint; follows the tunnel URL when it changes |
| `kis bench` | throughput at each concurrency level |
| `kis logs` | follow server events |
| `kis env` | print `OPENAI_*` exports for the direct endpoint |
| `kis down` | stop the server |
| `kis build [--ref vX.Y.Z]` | compile llama.cpp on Kaggle |
| `kis calibrate [model...]` | measure on Kaggle how many slots and how much context each preset fits |

`parallel` is the number of slots per llama-server instance (2 replicas → 2× the slots) and `ctx` the context each slot is guaranteed. If they don't fit in VRAM, the server starts with fewer slots, never less context, and `backends_ready` shows what it got.

## Models

Two example configs share the same presets and differ only in slots and context per slot. Both keep the KV cache in f16, so quality isn't reduced. The numbers are measured on Kaggle's T4s with `kis calibrate`.

- `config.example.toml`: **long context.** 4 slots, each with the largest context that fits, capped at the model's trained context.
- `config.throughput.example.toml`: **many slots.** As many slots as fit at 32K context each.

| preset | layout | long context: slots × ctx | throughput: slots × ctx |
| --- | --- | --- | --- |
| `gemma-4-e2b` | 2 replicas | 4 × 128K | 64 × 32K |
| `gemma-4-e4b` | 2 replicas | 4 × 128K | 42 × 32K |
| `qwen35-4b` | 2 replicas | 4 × 128K | 18 × 32K |
| `qwen35-9b` | 2 replicas | 4 × 112K | 14 × 32K |
| `gemma-4-e4b-q8` | 2 replicas | 4 × 128K | 34 × 32K |
| `qwen35-4b-q8` | 2 replicas | 4 × 128K | 14 × 32K |
| `qwen35-9b-q8` | 2 replicas | 4 × 80K | 8 × 32K |
| `gemma-4-12b` | 2 replicas | 4 × 192K | 14 × 32K |
| `gemma-4-26b-a4b` | split | 4 × 128K | 14 × 32K |
| `qwen38-27b` | split | 4 × 40K | 4 × 32K |

Gemma 4 E2B and E4B are trained for 128K, so their long-context limit is the model's own. Each preset pins a revision and sha256 and uses MTP speculative decoding. `kis up <model> --parallel N --ctx N` overrides a preset for one run. After a llama.cpp update or for new presets, rerun `kis calibrate [model...]`.

## Tunnels

Set `[tunnel] kind` in `config.toml`:

- **`cloudflared`** (default): no account needed, random URL per session. Streaming (SSE) works in practice, although Cloudflare doesn't document it for quick tunnels.
- **`cloudflared` + `token` + `public_url`**: a named tunnel with a stable hostname on your Cloudflare domain. Point it to `http://localhost:8080`.
- **`tailscale` + `authkey`**: private tailnet hostname (use a reusable, ephemeral key).

Cloudflare cuts a request whose response hasn't started within 100 s, such as a long prompt without streaming. If a reply hasn't started after 30 s, the server sends a 200 status and keeps the connection alive with whitespace (JSON) or comments (SSE). An error that comes after that point arrives with status 200.

## Docker

```bash
cp config.example.toml config.toml && mkdir -p .kis
docker compose run --rm kis up qwen35-4b
docker compose up -d     # proxy on 127.0.0.1:8080
```

Image: `ghcr.io/nasus20202/kaggleinferenceserver`.

## Development

```bash
uv run pytest && uv run ruff format . && uv run ruff check .
```

CI runs these checks, then builds the image and publishes it to GHCR from `main`. Renovate automerges minor and patch updates and checks weekly for new llama.cpp releases. CI doesn't build llama.cpp, so after a llama.cpp update, run `kis build` again.

`kaggle/` holds the scripts that run on Kaggle; `kis/` is the local CLI. Running a public endpoint from Kaggle is a grey area under its terms, so keep the kernel private (the default) and for personal use.
