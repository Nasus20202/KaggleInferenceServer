"""Typed settings and messages shared by kis and the Kaggle scripts.

kis builds the settings from config.toml and sends them to a kernel as plain data
(the rendered script's params block); the script reads them back with `load`, which
checks every key and type. Events and stats travel as JSON the same way. Kaggle runs
a single file, so `kis` pastes this module over the scripts' import of it: it may
only use the standard library, and Python 3.11.
"""

import dataclasses
import enum
import types
import typing
from dataclasses import dataclass, field
from typing import Any, Literal, TypeVar

T = TypeVar("T")

DEFAULT_NTFY_SERVER = "https://ntfy.sh"
CALIBRATE_TOPIC_SUFFIX = "-calibrate"  # calibration events are kept apart from the server's


class SchemaError(ValueError):
    """Data with an unknown key, a missing key or a value of the wrong type."""


# --------------------------------------------------------------------------- (de)serialization


def load(cls: type[T], data: object, where: str = "", strict: bool = True) -> T:
    """Build dataclass `cls` from `data` (parsed TOML or JSON), checking keys and types.
    `strict=False` ignores unknown keys (state written by an older kis)."""
    where = where or cls.__name__
    if not isinstance(data, dict):
        raise SchemaError(f"{where}: expected a table, got {type(data).__name__}")
    hints = typing.get_type_hints(cls)
    known = {f.name for f in dataclasses.fields(cls)}  # type: ignore[arg-type]
    if strict and (unknown := sorted(data.keys() - known)):
        raise SchemaError(f"{where}: unknown key(s) {', '.join(unknown)}")
    values = {k: _convert(hints[k], v, f"{where}.{k}", strict) for k, v in data.items() if k in known}
    try:
        return cls(**values)
    except TypeError as e:  # a required key is missing
        raise SchemaError(f"{where}: {e}") from None


def dump(obj: object) -> Any:
    """Dataclasses (and enums) as plain data, for JSON, a params block or a state file."""
    if dataclasses.is_dataclass(obj) and not isinstance(obj, type):
        return {f.name: dump(getattr(obj, f.name)) for f in dataclasses.fields(obj)}
    if isinstance(obj, enum.Enum):
        return obj.value
    if isinstance(obj, list | tuple):
        return [dump(v) for v in obj]
    if isinstance(obj, dict):
        return {k: dump(v) for k, v in obj.items()}
    return obj


def _convert(tp: Any, value: object, where: str, strict: bool) -> Any:
    origin, args = typing.get_origin(tp), typing.get_args(tp)
    if tp is Any:
        return value
    if dataclasses.is_dataclass(tp):
        return load(tp, value, where, strict)  # type: ignore[arg-type]
    if isinstance(tp, type) and issubclass(tp, enum.Enum):
        try:
            return tp(value)
        except ValueError:
            options = ", ".join(repr(m.value) for m in tp)
            raise SchemaError(f"{where}: {value!r} is not one of {options}") from None
    if origin in (typing.Union, types.UnionType):
        if value is None and type(None) in args:
            return None
        errors = []
        for option in (a for a in args if a is not type(None)):
            try:
                return _convert(option, value, where, strict)
            except SchemaError as e:
                errors.append(str(e))
        raise SchemaError(errors[0] if len(errors) == 1 else f"{where}: {value!r} matches none of {args}")
    if origin is Literal:
        if value not in args:
            raise SchemaError(f"{where}: {value!r} is not one of {', '.join(map(repr, args))}")
        return value
    if origin is list:
        if not isinstance(value, list):
            raise SchemaError(f"{where}: expected a list, got {type(value).__name__}")
        return [_convert(args[0], v, f"{where}[{i}]", strict) for i, v in enumerate(value)]
    if origin is dict:
        if not isinstance(value, dict):
            raise SchemaError(f"{where}: expected a table, got {type(value).__name__}")
        return {k: _convert(args[1], v, f"{where}.{k}", strict) for k, v in value.items()}
    if tp is float and isinstance(value, int) and not isinstance(value, bool):
        return float(value)
    if isinstance(tp, type) and (not isinstance(value, tp) or (tp is int and isinstance(value, bool))):
        raise SchemaError(f"{where}: expected {tp.__name__}, got {value!r}")
    return value


# --------------------------------------------------------------------------- settings


class Topology(enum.StrEnum):
    REPLICAS = "replicas"  # one llama-server per GPU
    SPLIT = "split"  # one llama-server across all GPUs


class TunnelKind(enum.StrEnum):
    CLOUDFLARED = "cloudflared"
    TAILSCALE = "tailscale"


@dataclass(frozen=True)
class Ntfy:
    """An ntfy server (ntfy.sh or self-hosted) and the access token for protected topics."""

    server: str = DEFAULT_NTFY_SERVER
    token: str = ""

    def url(self, topic: str) -> str:
        return f"{self.server.rstrip('/')}/{topic}"

    def headers(self) -> dict[str, str]:
        return {"Authorization": f"Bearer {self.token}"} if self.token else {}


@dataclass(frozen=True)
class Notify:
    """Optional human-readable notifications (phone app); never the endpoint or keys."""

    topic: str = ""
    token: str = ""
    server: str = ""  # default: the [ntfy] server

    def via(self, default: Ntfy) -> Ntfy:
        """The server to post to; on the [ntfy] server, its token unless this one has its own."""
        server = self.server or default.server
        same = server.rstrip("/") == default.server.rstrip("/")
        return Ntfy(server, self.token or (default.token if same else ""))


@dataclass(frozen=True)
class Tunnel:
    kind: TunnelKind = TunnelKind.CLOUDFLARED
    token: str = ""  # cloudflared named tunnel token
    public_url: str = ""  # named tunnel hostname, e.g. https://llm.example.com
    authkey: str = ""  # tailscale auth key
    hostname: str = "kaggle-llm"


@dataclass(frozen=True)
class Model:
    """A model preset: GGUF files on Hugging Face and how to serve them."""

    repo: str
    file: str
    alias: str  # model name the API reports
    parallel: int  # slots per llama-server instance
    revision: str | None = None
    sha256: str | None = None
    draft_file: str | None = None  # MTP draft model in the same repo
    draft_sha256: str | None = None
    ctx: int | None = None  # tokens per slot; default: server.ctx
    topology: Topology | None = None  # default: by file size and replica_max_gb
    tensor_split: str | None = None  # split topology: layers per GPU, e.g. "5,4"
    args: list[str] = field(default_factory=list)


@dataclass
class ServerConfig:
    """Everything kaggle/server.py needs; `kis up` sends it as the CONFIG params."""

    api_key: str
    model: Model  # loaded first
    preset: str = ""  # its preset name
    models: dict[str, Model] = field(default_factory=dict)  # presets the session can swap to, `model` included
    autoload: bool = True  # a request naming another preset swaps to it
    prefetch: list[str] = field(default_factory=list)  # presets downloaded in the background after startup
    resident: list[str] = field(default_factory=list)  # presets served on GPU 0 for the whole session, never swapped
    ntfy: Ntfy = field(default_factory=Ntfy)
    ntfy_topic: str = ""  # private control topic: events and the endpoint
    session: str = ""  # id in every event; `kis` tells sessions apart during a rollover
    notify: Notify = field(default_factory=Notify)
    tunnel: Tunnel = field(default_factory=Tunnel)
    idle_minutes: float = 10
    max_hours: float = 11.5
    replica_max_gb: float = 11.0
    ctx: int = 32768
    max_queue: int | None = None  # requests waiting beyond the slots before 429; None = no limit
    sticky_slack: int = 4  # see `pick` in kaggle/balancer.py; -1 = always the least busy instance
    args: list[str] = field(default_factory=list)


@dataclass
class CalibrateConfig:
    """Everything kaggle/calibrate.py needs; `kis calibrate` sends it as the CONFIG params."""

    run: str  # id of this run in its events
    models: dict[str, Model]
    args: list[str] = field(default_factory=list)
    ntfy: Ntfy = field(default_factory=Ntfy)
    ntfy_topic: str = ""
    replica_max_gb: float = 11.0


# --------------------------------------------------------------------------- events


class EventType(enum.StrEnum):
    # server sessions
    STARTING = "starting"
    DOWNLOADED = "downloaded"
    RETRY = "retry"  # out of memory: fewer slots
    BACKENDS_READY = "backends_ready"
    READY = "ready"  # with the endpoint
    STATS = "stats"  # heartbeat
    TUNNEL_RESTART = "tunnel_restart"
    STOPPED = "stopped"
    ERROR = "error"
    MODEL_LOADING = "model_loading"  # swapping to another preset
    MODEL_READY = "model_ready"
    MODEL_FAILED = "model_failed"  # the session goes on with the previous preset
    # calibration runs
    CALIBRATING = "calibrating"
    CALIBRATED = "calibrated"
    CALIBRATE_FAILED = "calibrate_failed"
    CALIBRATION_DONE = "calibration_done"


ENDED = frozenset({EventType.STOPPED, EventType.ERROR})  # a session that sent one of these is gone


@dataclass(frozen=True)
class Event:
    """A status message on the control topic. On the wire it is one flat JSON object:
    {"event": type, "t": ..., "session": ..., "run": ..., **data}."""

    type: EventType
    t: int = 0  # seconds since the kernel started
    session: str = ""  # server session id
    run: str = ""  # calibration run id
    data: dict[str, Any] = field(default_factory=dict)  # the event's own fields
    at: float = 0.0  # unix time ntfy received it (set by kis, not sent)

    HEADER: typing.ClassVar = ("event", "t", "session", "run")

    def to_json(self) -> dict[str, Any]:
        head = {"event": self.type.value, "t": self.t, "session": self.session, "run": self.run}
        return {**{k: v for k, v in head.items() if v != ""}, **dump(self.data)}

    @classmethod
    def from_json(cls, msg: dict[str, Any], at: float = 0.0) -> "Event":
        """Raises ValueError for an unknown event type (or not an event at all)."""
        data = {k: v for k, v in msg.items() if k not in cls.HEADER}
        return cls(EventType(msg["event"]), int(msg.get("t", 0)), msg.get("session", ""), msg.get("run", ""), data, at)

    @property
    def endpoint(self) -> str | None:
        return self.data.get("endpoint")

    @property
    def problem(self) -> str:
        """Why the session failed or stopped."""
        return str(self.data.get("error") or self.data.get("reason") or "")


# --------------------------------------------------------------------------- server API


class Route(enum.StrEnum):
    """Routes of the Kaggle-side balancer besides the OpenAI API (which it forwards)."""

    HEALTH = "/health"  # no API key needed
    STATS = "/admin/stats"  # GET: Stats as JSON
    LOGS = "/admin/logs"  # GET ?file=server.log&lines=200
    SHUTDOWN = "/admin/shutdown"  # POST ?session=<id>: 409 if that is another session
    LOAD = "/admin/load"  # POST ?model=<preset or alias>: 202 while it loads, 200 if loaded
    MODELS = "/v1/models"  # every preset of the session, with its status (also at /models)


class ModelStatus(enum.StrEnum):
    """A preset's state in /v1/models."""

    LOADED = "loaded"
    LOADING = "loading"
    DOWNLOADED = "downloaded"  # on disk, loads without a download
    AVAILABLE = "available"


SERVER_LOG = "server.log"


def api_error(message: str, **fields: str) -> dict[str, Any]:
    """An error body as the OpenAI API sends it, so clients show the message."""
    return {"error": {"message": message, **fields}}


@dataclass(frozen=True)
class GpuStats:
    gpu: int
    used_mib: int
    total_mib: int
    util_pct: int


@dataclass(frozen=True)
class CpuStats:
    """CPU use of the Kaggle container since the previous reading."""

    cores: float  # CPUs the container may use
    used_pct: int  # of all its cores
    processes: dict[str, int] = field(default_factory=dict)  # llama-server port -> % of one core


@dataclass(frozen=True)
class RamStats:
    """Host memory of the Kaggle container, where llama-server keeps its prompt cache."""

    used_mib: int  # without the page cache, which the kernel drops under pressure
    total_mib: int  # the container's limit, or the machine's memory if it has none
    processes: dict[str, int] = field(default_factory=dict)  # llama-server port -> anonymous RSS (MiB)


@dataclass(frozen=True)
class UsageSummary:
    """Totals of the completions of one model since the session started."""

    requests: int
    tokens_in: int
    tokens_out: int
    tokens_cached: int
    cache_rate: float | None
    prefill_tps: float | None
    decode_tps: float | None
    draft_acceptance: float | None


@dataclass(frozen=True, kw_only=True)
class Stats:
    """What the server reports, in the order it reads best: the session, its load, then the
    hardware (GPUs, CPU, RAM)."""

    session: str
    preset: str = ""  # name of the loaded preset
    model: str  # alias of the loaded preset
    topology: Topology
    slots: int  # parallel x instances
    parallel: int  # slots per instance
    ctx: int  # tokens per slot
    resident: list[str] = field(default_factory=list)  # aliases of the resident presets
    uptime_s: int
    idle_s: int
    requests: int
    inflight: int
    errors: int
    usage: dict[str, UsageSummary]  # per model name
    gpus: list[GpuStats]
    cpu: CpuStats | None = None
    ram: RamStats | None = None


# --------------------------------------------------------------------------- calibration

PROFILE_CTX = {"32k": 32768, "64k": 65536, "96k": 98304, "128k": 131072}  # tokens per slot of each profile
MAX_PROFILE = "max"  # the largest context per slot that fits


@dataclass
class Fit:
    """What one calibration profile fits, and how fast it runs."""

    topology: Topology
    instances: int
    parallel: int  # slots per instance
    ctx: int  # tokens per slot
    used_mib: dict[str, int] = field(default_factory=dict)  # per GPU index
    single_tps: float | None = None  # one request alone
    total_tps: float | None = None  # all slots busy

    @property
    def slots(self) -> int:
        return self.parallel * self.instances


@dataclass(frozen=True)
class Calibration:
    """Result of `kis calibrate` for one model: the data of its `calibrated` event."""

    model: str
    model_gb: float
    n_ctx_train: int
    profiles: dict[str, Fit]  # per profile name (32k, ..., max)
