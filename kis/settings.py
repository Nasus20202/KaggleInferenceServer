"""config.toml, generated secrets and local state (.kis/)."""

import dataclasses
import json
import os
import secrets
import sys
import tomllib
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

from .schema import Calibration, Model, Notify, Ntfy, SchemaError, ServerConfig, Tunnel, dump, load

ROOT = Path(__file__).resolve().parent.parent
STATE_DIR = ROOT / ".kis"


@dataclass(frozen=True)
class Kaggle:
    username: str
    llama_cpp_ref: str
    accelerator: str = "NvidiaTeslaT4"


@dataclass
class Server:
    idle_minutes: float = 10
    max_hours: float = 11.5
    rollover_hours: float = 0  # 0 = off
    max_queue: int | None = None
    replica_max_gb: float = 11.0
    ctx: int = 32768
    autoload: bool = True
    prefetch: list[str] = field(default_factory=list)
    args: list[str] = field(default_factory=list)


@dataclass
class Config:
    """config.toml."""

    kaggle: Kaggle
    models: dict[str, Model]
    tunnel: Tunnel = field(default_factory=Tunnel)
    ntfy: Ntfy = field(default_factory=Ntfy)
    notify: Notify = field(default_factory=Notify)
    server: Server = field(default_factory=Server)


@dataclass(frozen=True)
class Secrets:
    """Generated on first use: the server's API key and kis' private control topic."""

    api_key: str
    ntfy_topic: str


@dataclass
class State:
    """.kis/state.json: the last session started and the calibration results."""

    server: ServerConfig | None = None  # settings of the last `kis up`, reused by a rollover
    slug: str = ""  # kernel of the last session
    session: str = ""
    model: str = ""  # alias of its loaded preset
    calibration: dict[str, Calibration] = field(default_factory=dict)  # per preset name


def parse_config(data: dict[str, Any], environ: dict[str, str] | None = None) -> Config:
    """Config from parsed TOML; KIS_NTFY_TOKEN and KIS_NOTIFY_TOKEN override the tokens."""
    config = load(Config, data, "config")
    env = os.environ if environ is None else environ
    if token := env.get("KIS_NTFY_TOKEN"):
        config.ntfy = dataclasses.replace(config.ntfy, token=token)
    if token := env.get("KIS_NOTIFY_TOKEN"):
        config.notify = dataclasses.replace(config.notify, token=token)
    return config


def load_config() -> Config:
    """config.toml, or the file named by KIS_CONFIG."""
    path = Path(os.environ.get("KIS_CONFIG") or ROOT / "config.toml")
    if not path.exists():
        sys.exit(f"{path} missing: cp examples/config.32k.toml config.toml and set kaggle.username")
    try:
        config = parse_config(tomllib.loads(path.read_text()))
    except (tomllib.TOMLDecodeError, SchemaError) as e:
        sys.exit(f"{path}: {e}")
    if config.kaggle.username == "your-kaggle-username":
        sys.exit("set kaggle.username in config.toml")
    return config


def _read(name: str) -> dict[str, Any]:
    path = STATE_DIR / name
    return json.loads(path.read_text()) if path.exists() else {}


def _write(name: str, data: object) -> None:
    STATE_DIR.mkdir(exist_ok=True)
    path = STATE_DIR / name
    path.write_text(json.dumps(data, indent=2))
    path.chmod(0o600)


def load_secrets() -> Secrets:
    """API key and private ntfy topic, generated on first use."""
    data = _read("secrets.json")
    if not data:
        data = {"api_key": "sk-kis-" + secrets.token_urlsafe(24), "ntfy_topic": "kis-" + secrets.token_hex(12)}
        _write("secrets.json", data)
    return load(Secrets, data, ".kis/secrets.json")


def load_state() -> State:
    """Saved state; keys of older kis versions are ignored, unreadable state starts over."""
    try:
        return load(State, _read("state.json"), ".kis/state.json", strict=False)
    except SchemaError as e:
        print(f"! ignoring {e}")
        return State()


def save_state(state: State) -> None:
    _write("state.json", dump(state))
