"""kis: llama.cpp on free Kaggle GPUs, used as a local OpenAI-compatible API.

kis build              compile llama-server for T4 once
kis up qwen35-9b       start the server kernel and wait for its endpoint
kis use gemma-4-12b    swap the running session to another preset (no preset: list them)
kis proxy              local endpoint http://127.0.0.1:8080/v1 (any API key)
kis bench              aggregate tok/s vs concurrency
kis calibrate          measure slots, context and speed of each preset on the T4s
kis status             GPU quota, kernels, VRAM and load of the running server
kis rollover           replace the running session without downtime (kis proxy does it on its own)
kis logs | env | down
"""

import argparse
import dataclasses
import logging
import os
import sys
import time
from http import HTTPStatus

from . import bench, events, kaggle, profiles, proxy, rollover, status
from .client import AdminClient
from .schema import (
    CALIBRATE_TOPIC_SUFFIX,
    ENDED,
    CalibrateConfig,
    Calibration,
    EventType,
    ServerConfig,
    Topology,
    TunnelKind,
    load,
)
from .settings import Config, Secrets, load_config, load_secrets, load_state, save_state

PROXY_PORT = 8080
KERNEL_CHECK_S = 60  # how often to look at the kernel state while waiting
EVENT_POLL_S = 5
SPEC_OPTIONS = ("--spec-type", "--spec-draft-n-max")  # speculative decoding, dropped by --no-spec
SLOT_OPTIONS = ("--kv-unified-per-slot", "--parallel")  # set by calibration itself


def cmd_build(args: argparse.Namespace, config: Config) -> None:
    ref = args.ref or config.kaggle.llama_cpp_ref
    kaggle.push(config, kaggle.BUILD, kaggle.BUILD_SCRIPT, {kaggle.LLAMA_CPP_REF_PARAM: ref})
    print(f"building llama.cpp {ref} (~25 min): {kaggle.kernel_url(config, kaggle.BUILD)}")
    while True:
        time.sleep(KERNEL_CHECK_S)
        state = kaggle.state(config, kaggle.BUILD)
        print(time.strftime("%H:%M"), state)
        if state == kaggle.KernelState.COMPLETE:
            return
        if state.ended:
            sys.exit("build failed, see the kernel log")


def server_config(args: argparse.Namespace, config: Config, secrets: Secrets) -> ServerConfig:
    """Settings of a server session: it starts with the preset `args.model`, with the command
    line's overrides, and can swap to every other preset of config.toml."""
    if args.model not in config.models:
        sys.exit(f"unknown model {args.model!r}; choose from: {', '.join(config.models)}")
    for key in ("prefetch", "resident"):
        if unknown := [n for n in getattr(config.server, key) if n not in config.models]:
            sys.exit(f"server.{key}: unknown models {', '.join(unknown)}")
    if args.model in config.server.resident:
        sys.exit(f"{args.model} is resident: start a preset that can be swapped")
    model = config.models[args.model]
    if args.parallel:
        model = dataclasses.replace(model, parallel=args.parallel)
    if args.topology:
        model = dataclasses.replace(model, topology=args.topology)
    if args.ctx:
        model = dataclasses.replace(model, ctx=args.ctx)
    if args.no_spec:
        model = dataclasses.replace(model, draft_file=None, args=_drop_options(model.args, *SPEC_OPTIONS))
    if config.tunnel.kind == TunnelKind.TAILSCALE and not config.tunnel.authkey:
        sys.exit("tunnel.kind = tailscale needs tunnel.authkey")
    server = config.server
    return ServerConfig(
        api_key=secrets.api_key,
        model=model,
        preset=args.model,
        models={**config.models, args.model: model},
        autoload=server.autoload,
        prefetch=[n for n in server.prefetch if n != args.model and n not in server.resident],
        resident=server.resident,
        ntfy=config.ntfy,
        ntfy_topic=secrets.ntfy_topic,
        notify=config.notify,
        tunnel=config.tunnel,
        idle_minutes=server.idle_minutes,
        max_hours=server.max_hours,
        replica_max_gb=server.replica_max_gb,
        ctx=server.ctx,
        max_queue=server.max_queue,
        sticky_slack=server.sticky_slack,
        args=server.args,
    )


def _drop_options(args: list[str], *names: str) -> list[str]:
    """Remove `--name value` pairs."""
    out: list[str] = []
    skip = False
    for a in args:
        if skip:
            skip = False
        elif a in names:
            skip = True
        else:
            out.append(a)
    return out


def cmd_up(args: argparse.Namespace, config: Config) -> None:
    secrets = load_secrets()
    server = server_config(args, config, secrets)
    if (current := events.current(config, secrets.ntfy_topic)) and not args.force:
        if args.parallel or args.topology or args.ctx or args.no_spec:
            sys.exit(
                f"a server is already up at {current.endpoint}; `kis down` first (or --force) to change its settings"
            )
        swap(config, secrets, current, args.model)  # same as `kis use`
        return
    q = status.quota()
    print(status.format_quota(q))
    if q and q.left_h < 0.5:
        print("! less than 30 min of GPU quota left; the session stops 2 min before it runs out")
    since = str(int(time.time()))
    slug = load_state().slug or kaggle.SERVER
    session = rollover.launch(config, server, slug)
    print(f"kernel: {kaggle.kernel_url(config, slug)}  session: {session}")
    follow(config, secrets.ntfy_topic, since, until_ready=True, session=session, slug=slug)


def follow(
    config: Config,
    topic: str,
    since: str,
    until_ready: bool = False,
    session: str | None = None,
    slug: str = "",
) -> None:
    """Print events (of one session, if given); with until_ready, return on `ready` and exit on failure."""
    last_status_check, last_state = time.time(), None
    while True:
        for msg_id, event in events.fetch(config.ntfy, topic, since):
            since = msg_id
            if session and event.session != session:
                continue
            events.show(event)
            if until_ready and event.type == EventType.READY:
                print(f"\nready: {event.endpoint}/v1\nlocal: `kis proxy`, then http://127.0.0.1:{PROXY_PORT}/v1")
                return
            if until_ready and event.type in ENDED:
                sys.exit(1)
        if until_ready and time.time() - last_status_check > KERNEL_CHECK_S:  # crashes before the first event
            last_status_check = time.time()
            state = kaggle.state(config, slug or kaggle.SERVER)
            if state.ended:
                sys.exit(f"kernel ended: {state}")
            if state != last_state:  # e.g. queued for a GPU, which no event reports
                print(f"kernel: {state}")
                last_state = state
        time.sleep(EVENT_POLL_S)


def cmd_use(args: argparse.Namespace, config: Config) -> None:
    secrets = load_secrets()
    current = events.current(config, secrets.ntfy_topic)
    if not args.model:
        list_models(config, secrets, current)
    elif not current:
        sys.exit(f"no running server; `kis up {args.model}`")
    else:
        swap(config, secrets, current, args.model)


def list_models(config: Config, secrets: Secrets, current: events.Session | None) -> None:
    """The running session's presets and their status, or config.toml's when none runs."""
    if current and current.endpoint:
        try:
            print(status.format_models(AdminClient(current.endpoint, secrets.api_key).models()))
            return
        except (OSError, RuntimeError) as e:
            print(f"! server: {e}; presets in config.toml:")
    else:
        print("no running server; presets in config.toml (`kis up <preset>`):")
    presets = [
        {"preset": name, "id": m.alias, "context_length": m.ctx or config.server.ctx, "parallel": m.parallel}
        for name, m in config.models.items()
    ]
    print(status.format_models(presets))


def swap(config: Config, secrets: Secrets, current: events.Session, model: str) -> None:
    """Swap the running session to preset `model` and print its events until it serves."""
    assert current.endpoint
    since = str(int(time.time()))
    code, answer = AdminClient(current.endpoint, secrets.api_key).load(model)
    if code == HTTPStatus.OK:
        print(f"{answer['model']} is already loaded")
    elif code == HTTPStatus.ACCEPTED:
        session = answer.get("session") or current.session  # a named tunnel may reach the other session
        print(f"loading {answer['model']} in session {session}")
        wait_loaded(config, secrets.ntfy_topic, since, session, answer["preset"])
    else:
        sys.exit((answer.get("error") or {}).get("message") or f"HTTP {code.value} {code.phrase}")
    state = load_state()
    state.model = answer["model"]
    save_state(state)


def wait_loaded(config: Config, topic: str, since: str, session: str, preset: str) -> None:
    """Print the events of `session` until `preset` is loaded; exit if loading it fails."""
    while True:
        for msg_id, event in events.fetch(config.ntfy, topic, since):
            since = msg_id
            if event.session != session:
                continue
            events.show(event)
            if event.data.get("preset") == preset and event.type == EventType.MODEL_READY:
                return
            if (event.data.get("preset") == preset and event.type == EventType.MODEL_FAILED) or event.type in ENDED:
                sys.exit(f"loading {preset} failed: {event.problem}")
        time.sleep(EVENT_POLL_S)


def cmd_rollover(args: argparse.Namespace, config: Config) -> None:
    secrets = load_secrets()
    old, url = rollover.rollover(config, secrets.ntfy_topic)
    print(f"new session ready: {url}/v1; draining session {old.session}")
    assert old.endpoint
    rollover.retire(old.endpoint, secrets.api_key, old.session)
    print("done")


def cmd_calibrate(args: argparse.Namespace, config: Config) -> None:
    names = args.models or list(config.models)
    unknown = [n for n in names if n not in config.models]
    if unknown:
        sys.exit(f"unknown models: {', '.join(unknown)}")
    topic = load_secrets().ntfy_topic + CALIBRATE_TOPIC_SUFFIX
    since = str(int(time.time()))
    run = CalibrateConfig(
        run=since,
        models={n: config.models[n] for n in names},
        args=_drop_options(config.server.args, *SLOT_OPTIONS),
        ntfy=config.ntfy,
        ntfy_topic=topic,
        replica_max_gb=config.server.replica_max_gb,
    )
    params = {kaggle.CONFIG_PARAM: run}
    kaggle.push(config, kaggle.CALIBRATE, kaggle.CALIBRATE_SCRIPT, params, sources=[kaggle.build_source(config)])
    print(f"kernel: {kaggle.kernel_url(config, kaggle.CALIBRATE)} (~10 min per model)")
    while True:
        for msg_id, event in events.fetch(config.ntfy, topic, since):
            since = msg_id
            if event.run != run.run:  # an older kernel version still running
                continue
            events.show(event)
            if event.type == EventType.CALIBRATED:
                result = load(Calibration, event.data, "calibrated event", strict=False)
                state = load_state()
                state.calibration[result.model] = result
                save_state(state)
            if event.type in (EventType.CALIBRATION_DONE, EventType.ERROR):
                results = load_state().calibration
                print(profiles.table(results))
                if args.write:
                    profiles.write(results)
                    print("updated examples/ and the README table")
                return
        time.sleep(EVENT_POLL_S)


def cmd_logs(args: argparse.Namespace, config: Config) -> None:
    secrets = load_secrets()
    if not args.file:
        follow(config, secrets.ntfy_topic, args.since)
        return
    url = events.endpoint(config, secrets.ntfy_topic)
    if not url:
        sys.exit("no running server")
    try:
        print(AdminClient(url, secrets.api_key, timeout=60).logs(args.file, args.lines), end="")
    except (RuntimeError, OSError) as e:  # an error answer, or the server can't be reached
        sys.exit(f"cannot read {args.file} from {url}: {e}")


def cmd_status(args: argparse.Namespace, config: Config) -> None:
    print(status.format_quota(status.quota()))
    for slug in (*rollover.SLUGS, kaggle.BUILD, kaggle.CALIBRATE):
        print(f"{slug}: {kaggle.state(config, slug)}")
    secrets = load_secrets()
    url = events.endpoint(config, secrets.ntfy_topic)
    if not url:
        print("server: not running")
        return
    print(f"endpoint: {url}/v1")
    try:
        stats = AdminClient(url, secrets.api_key).stats()
    except OSError as e:
        stats, error = None, str(e)
    else:
        error = "the server answered with an error"
    print(status.format_stats(stats) if stats else f"stats unavailable: {error}")


def cmd_down(args: argparse.Namespace, config: Config) -> None:
    secrets = load_secrets()
    current = events.current(config, secrets.ntfy_topic)
    if not current or not current.endpoint:
        sys.exit("no running server")
    client = AdminClient(current.endpoint, secrets.api_key)
    code = client.shutdown(current.session)
    for _ in range(9):  # a named tunnel shared by two sessions may route to the other one: ask again
        if code != HTTPStatus.CONFLICT:
            break
        code = client.shutdown(current.session)
    if code != HTTPStatus.OK:
        sys.exit(f"shutdown of session {current.session} failed: HTTP {code.value} {code.phrase}")
    print(f"session {current.session} is stopping")


def cmd_env(args: argparse.Namespace, config: Config) -> None:
    secrets = load_secrets()
    url = events.endpoint(config, secrets.ntfy_topic)
    print(f"export OPENAI_BASE_URL={url or '<not running>'}/v1")
    print(f"export OPENAI_API_KEY={secrets.api_key}")


def cmd_proxy(args: argparse.Namespace, config: Config) -> None:
    proxy.run(config, load_secrets(), args.host, args.port)


def cmd_bench(args: argparse.Namespace, config: Config) -> None:
    secrets = load_secrets()
    url = args.url or events.endpoint(config, secrets.ntfy_topic)
    if not url:
        sys.exit("no running server")
    levels = [int(c) for c in args.concurrency.split(",")]
    try:  # the loaded preset: naming another one would swap to it
        stats = AdminClient(url, secrets.api_key).stats()
    except OSError:
        stats = None
    model = stats.model if stats else load_state().model or "model"
    bench.run(url, secrets.api_key, model, levels, args.max_tokens, args.min_requests)


def main() -> None:
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
    p.add_argument("--topology", type=Topology, choices=list(Topology))
    p.add_argument("--ctx", type=int, help="context per slot in tokens")
    p.add_argument("--no-spec", action="store_true", help="disable MTP speculative decoding")
    p.add_argument("--force", action="store_true", help="start a new session even if one is up")

    p = sub.add_parser("use", help="swap the running session to another preset; without one, list them")
    p.add_argument("model", nargs="?", help="preset name or alias")

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
    p.add_argument("--port", type=int, default=PROXY_PORT)

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
        globals()[f"cmd_{args.cmd}"](args, load_config(required=args.cmd != "proxy"))
    except (kaggle.KaggleError, rollover.RolloverError) as e:
        sys.exit(str(e))
