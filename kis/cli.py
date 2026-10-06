"""kis: llama.cpp on free Kaggle GPUs, used as a local OpenAI-compatible API.

kis build              compile llama-server for T4 once
kis up qwen35-9b       start the server kernel and wait for its endpoint
kis proxy              local endpoint http://127.0.0.1:8080/v1 (any API key)
kis bench              aggregate tok/s vs concurrency
kis logs | env | down
"""

import argparse
import sys
import time
import urllib.request

from . import bench, events, kaggle, proxy
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
            "replica_max_gb": server["replica_max_gb"],
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
    since = str(int(time.time()))
    kaggle.push(config, kaggle.SERVER, "server.py", params, sources=[kaggle.kernel_id(config, kaggle.BUILD)])
    save_state(model=params["CONFIG"]["model"]["alias"], since=since)
    print(f"kernel: {kaggle.kernel_url(config, kaggle.SERVER)}")
    follow(config, secrets["ntfy_topic"], since, until_ready=True)


def follow(config, topic: str, since: str, until_ready: bool = False):
    """Print events; with until_ready, return on `ready` and exit on failure."""
    last_status_check = time.time()
    while True:
        for msg_id, event in events.fetch(topic, since):
            since = msg_id
            events.show(event)
            if until_ready and event["event"] == "ready":
                print(f"\nready: {event['endpoint']}/v1\nlocal: `kis proxy`, then http://127.0.0.1:8080/v1")
                return
            if until_ready and event["event"] in ("error", "stopped"):
                sys.exit(1)
        if until_ready and time.time() - last_status_check > 60:  # catches crashes before the first event
            last_status_check = time.time()
            status = kaggle.status(config, kaggle.SERVER)
            if any(word in status.lower() for word in ("error", "cancel", "complete")):
                sys.exit(f"kernel ended: {status}")
        time.sleep(5)


def cmd_logs(args, config):
    follow(config, load_secrets()["ntfy_topic"], args.since)


def cmd_down(args, config):
    secrets = load_secrets()
    url = events.endpoint(config, secrets["ntfy_topic"])
    if not url:
        sys.exit("no running server")
    req = urllib.request.Request(
        f"{url}/admin/shutdown", method="POST", headers={"Authorization": f"Bearer {secrets['api_key']}"}
    )
    print(urllib.request.urlopen(req, timeout=30).read().decode())


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
    sub = parser.add_subparsers(dest="cmd", required=True)

    p = sub.add_parser("build", help="compile llama-server for T4 on Kaggle")
    p.add_argument("--ref", help="llama.cpp tag or branch (default: kaggle.llama_cpp_ref)")

    p = sub.add_parser("up", help="start the server kernel with a model preset")
    p.add_argument("model")
    p.add_argument("--parallel", type=int, help="slots per llama-server instance")
    p.add_argument("--topology", choices=["replicas", "split"])
    p.add_argument("--no-spec", action="store_true", help="disable MTP speculative decoding")
    p.add_argument("--force", action="store_true", help="push even if a server is already up")

    p = sub.add_parser("logs", help="follow server events")
    p.add_argument("--since", default="1h")
    sub.add_parser("down", help="stop the server now (saves GPU quota)")
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
    globals()[f"cmd_{args.cmd}"](args, load_config())
