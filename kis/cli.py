"""kis: llama.cpp on free Kaggle GPUs, used as a local OpenAI-compatible API.

kis build              compile llama-server for T4 once
kis up qwen35-9b       start the server kernel and wait for its endpoint
kis proxy              local endpoint http://127.0.0.1:8080/v1 (any API key)
kis bench              aggregate tok/s vs concurrency
kis calibrate          measure slots, context and speed of each preset on the T4s
kis status             GPU quota, kernels, VRAM and load of the running server
kis rollover           replace the running session without downtime (kis proxy does it on its own)
kis logs | env | down
"""

import argparse
import logging
import os
import sys
import time
import urllib.error
import urllib.request

from . import bench, events, kaggle, profiles, proxy, retry, rollover, status
from .settings import load_config, load_secrets, load_state, save_state


def cmd_build(args, config):
    ref = args.ref or config["kaggle"]["llama_cpp_ref"]
    kaggle.push(config, kaggle.BUILD, "build.py", {"LLAMA_CPP_REF": ref})
    print(f"building llama.cpp {ref} (~25 min): {kaggle.kernel_url(config, kaggle.BUILD)}")
    while True:
        time.sleep(60)
        status = kaggle.status(config, kaggle.BUILD)
        print(time.strftime("%H:%M"), status)
        if "complete" in status.lower():
            return
        if any(word in status.lower() for word in ("error", "cancel")):
            sys.exit("build failed, see the kernel log")


def server_params(args, config, secrets) -> dict:
    if args.model not in config["models"]:
        sys.exit(f"unknown model {args.model!r}; choose from: {', '.join(config['models'])}")
    model = dict(config["models"][args.model])
    if args.parallel:
        model["parallel"] = args.parallel
    if args.topology:
        model["topology"] = args.topology
    if args.ctx:
        model["ctx"] = args.ctx
    if args.no_spec:
        model["draft_file"] = None
        model["args"] = _drop_options(model.get("args", []), "--spec-type", "--spec-draft-n-max")
    tunnel = dict(config["tunnel"])
    if tunnel["kind"] == "tailscale" and not tunnel.get("authkey"):
        sys.exit("tunnel.kind = tailscale needs tunnel.authkey")
    server = config["server"]
    return {
        "CONFIG": {
            "api_key": secrets["api_key"],
            "ntfy_topic": secrets["ntfy_topic"],
            "tunnel": tunnel,
            "idle_minutes": server["idle_minutes"],
            "max_hours": server["max_hours"],
            "session": "",
            "notify": {
                "topic": config.get("notify", {}).get("topic", ""),
                "token": os.environ.get("KIS_NOTIFY_TOKEN") or config.get("notify", {}).get("token", ""),
            },
            "replica_max_gb": server["replica_max_gb"],
            "ctx": server["ctx"],
            "max_queue": server.get("max_queue"),
            "args": server["args"],
            "model": model,
        }
    }


def _drop_options(args: list[str], *names: str) -> list[str]:
    """Remove `--name value` pairs."""
    out, skip = [], False
    for a in args:
        if skip:
            skip = False
        elif a in names:
            skip = True
        else:
            out.append(a)
    return out


def cmd_up(args, config):
    secrets = load_secrets()
    params = server_params(args, config, secrets)
    if (url := events.endpoint(config, secrets["ntfy_topic"])) and not args.force:
        sys.exit(f"a server is already up at {url}; `kis down` first (or --force)")
    q = status.quota()
    print(status.format_quota(q))
    if q and q["total_h"] - q["used_h"] < 0.5:
        print("! less than 30 min of GPU quota left; the session stops 2 min before it runs out")
    since = str(int(time.time()))
    slug = load_state().get("slug", kaggle.SERVER)
    session = rollover.launch(config, params, slug)
    print(f"kernel: {kaggle.kernel_url(config, slug)}  session: {session}")
    follow(config, secrets["ntfy_topic"], since, until_ready=True, session=session, slug=slug)


def follow(config, topic: str, since: str, until_ready: bool = False, session: str | None = None, slug: str = ""):
    """Print events (of one session, if given); with until_ready, return on `ready` and exit on failure."""
    last_status_check = time.time()
    while True:
        for msg_id, event in events.fetch(topic, since):
            since = msg_id
            if session and event.get("session") != session:
                continue
            events.show(event)
            if until_ready and event["event"] == "ready":
                print(f"\nready: {event['endpoint']}/v1\nlocal: `kis proxy`, then http://127.0.0.1:8080/v1")
                return
            if until_ready and event["event"] in ("error", "stopped"):
                sys.exit(1)
        if until_ready and time.time() - last_status_check > 60:  # catches crashes before the first event
            last_status_check = time.time()
            state = kaggle.status(config, slug or kaggle.SERVER)
            if any(word in state.lower() for word in ("error", "cancel", "complete")):
                sys.exit(f"kernel ended: {state}")
        time.sleep(5)


def cmd_rollover(args, config):
    secrets = load_secrets()
    old, url = rollover.rollover(config, secrets["ntfy_topic"])
    print(f"new session ready: {url}/v1; draining session {old['session']}")
    rollover.retire(old["endpoint"], secrets["api_key"], old["session"])
    print("done")


def cmd_calibrate(args, config):
    names = args.models or list(config["models"])
    unknown = [n for n in names if n not in config["models"]]
    if unknown:
        sys.exit(f"unknown models: {', '.join(unknown)}")
    topic = load_secrets()["ntfy_topic"] + "-calibrate"  # kept apart from the server's events
    server = config["server"]
    since = str(int(time.time()))
    params = {
        "RUN": since,
        "MODELS": {n: config["models"][n] for n in names},
        "ARGS": _drop_options(server["args"], "--kv-unified-per-slot", "--parallel"),
        "NTFY_TOPIC": topic,
        "REPLICA_MAX_GB": server["replica_max_gb"],
    }
    kaggle.push(config, kaggle.CALIBRATE, "calibrate.py", params, sources=[kaggle.kernel_id(config, kaggle.BUILD)])
    print(f"kernel: {kaggle.kernel_url(config, kaggle.CALIBRATE)} (~10 min per model)")
    while True:
        for msg_id, event in events.fetch(topic, since):
            since = msg_id
            if event.get("run") != params["RUN"]:  # an older kernel version still running
                continue
            events.show(event)
            if event["event"] == "calibrated":
                save_state(calibration={**load_state().get("calibration", {}), event["model"]: event})
            if event["event"] in ("calibration_done", "error"):
                results = load_state().get("calibration", {})
                print(profiles.table(results))
                if args.write:
                    profiles.write(results)
                    print("updated examples/ and the README table")
                return
        time.sleep(10)


def cmd_logs(args, config):
    secrets = load_secrets()
    if not args.file:
        follow(config, secrets["ntfy_topic"], args.since)
        return
    url = events.endpoint(config, secrets["ntfy_topic"])
    if not url:
        sys.exit("no running server")
    req = urllib.request.Request(
        f"{url}/admin/logs?file={args.file}&lines={args.lines}",
        headers={"Authorization": f"Bearer {secrets['api_key']}"},
    )
    try:
        print(retry.urlopen(req, timeout=60, what="server log").read().decode(), end="")
    except urllib.error.HTTPError as e:
        sys.exit(e.read().decode())


def cmd_status(args, config):
    print(status.format_quota(status.quota()))
    for slug in (*rollover.SLUGS, kaggle.BUILD, kaggle.CALIBRATE):
        print(f"{slug}: {kaggle.state(config, slug)}")
    secrets = load_secrets()
    url = events.endpoint(config, secrets["ntfy_topic"])
    if not url:
        print("server: not running")
        return
    print(f"endpoint: {url}/v1")
    try:
        print(status.format_stats(status.server_stats(url, secrets["api_key"])))
    except OSError as e:
        print(f"stats unavailable: {e}")


def cmd_down(args, config):
    secrets = load_secrets()
    current = events.current(config, secrets["ntfy_topic"])
    if not current:
        sys.exit("no running server")
    for _ in range(10):  # a named tunnel shared by two sessions may route to the other one: ask again
        code, body = rollover.admin(
            current["endpoint"], secrets["api_key"], f"/admin/shutdown?session={current['session']}", "POST"
        )
        if code != 409:
            break
    print(body)


def cmd_env(args, config):
    secrets = load_secrets()
    url = events.endpoint(config, secrets["ntfy_topic"])
    print(f"export OPENAI_BASE_URL={url or '<not running>'}/v1")
    print(f"export OPENAI_API_KEY={secrets['api_key']}")


def cmd_proxy(args, config):
    proxy.run(config, load_secrets(), args.host, args.port)


def cmd_bench(args, config):
    secrets = load_secrets()
    url = args.url or events.endpoint(config, secrets["ntfy_topic"])
    if not url:
        sys.exit("no running server")
    levels = [int(c) for c in args.concurrency.split(",")]
    bench.run(url, secrets["api_key"], load_state().get("model", "model"), levels, args.max_tokens, args.min_requests)


def main():
    parser = argparse.ArgumentParser(
        prog="kis", description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter
    )
    parser.add_argument("-v", "--verbose", action="store_true", help="debug logging (or KIS_LOG_LEVEL=DEBUG)")
    sub = parser.add_subparsers(dest="cmd", required=True)

    p = sub.add_parser("build", help="compile llama-server for T4 on Kaggle")
    p.add_argument("--ref", help="llama.cpp tag or branch (default: kaggle.llama_cpp_ref)")

    p = sub.add_parser("up", help="start the server kernel with a model preset")
    p.add_argument("model")
    p.add_argument("--parallel", type=int, help="slots per llama-server instance")
    p.add_argument("--topology", choices=["replicas", "split"])
    p.add_argument("--ctx", type=int, help="context per slot in tokens")
    p.add_argument("--no-spec", action="store_true", help="disable MTP speculative decoding")
    p.add_argument("--force", action="store_true", help="push even if a server is already up")

    p = sub.add_parser("calibrate", help="measure slots, context and speed per preset on Kaggle")
    p.add_argument("models", nargs="*", help="presets to measure (default: all)")
    p.add_argument("--write", action="store_true", help="regenerate examples/*.toml and the README table")

    p = sub.add_parser("logs", help="follow server events, or print a server-side log file")
    p.add_argument("--since", default="1h", help="events since a duration or unix time")
    p.add_argument("--file", help="server.log or llama-8090.log, llama-8091.log (running server)")
    p.add_argument("--lines", type=int, default=200)
    sub.add_parser("status", help="GPU quota, kernel states, VRAM and load of the running server")
    sub.add_parser("down", help="stop the server now (saves GPU quota)")
    sub.add_parser("rollover", help="replace the running session with a fresh one, without downtime")
    sub.add_parser("env", help="print OPENAI_* exports for the direct endpoint")

    p = sub.add_parser("proxy", help="stable local endpoint")
    p.add_argument("--host", default="127.0.0.1", help="0.0.0.0 inside a container")
    p.add_argument("--port", type=int, default=8080)

    p = sub.add_parser("bench", help="throughput vs concurrency")
    p.add_argument("--concurrency", default="1,4,8,16,32")
    p.add_argument("--max-tokens", type=int, default=256)
    p.add_argument("--min-requests", type=int, default=8)
    p.add_argument("--url", help="endpoint base URL (default: current server)")

    args = parser.parse_args()
    level = "DEBUG" if args.verbose else os.environ.get("KIS_LOG_LEVEL", "INFO")
    logging.basicConfig(level=level.upper(), format="%(asctime)s %(levelname)s %(name)s: %(message)s")
    logging.getLogger("aiohttp.access").setLevel(logging.WARNING)  # the proxy logs its own request lines
    try:
        globals()[f"cmd_{args.cmd}"](args, load_config())
    except (kaggle.KaggleError, rollover.RolloverError) as e:
        sys.exit(str(e))
