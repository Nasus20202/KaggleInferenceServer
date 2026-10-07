"""config.toml, generated secrets and local state (.kis/)."""

import dataclasses
import json
import os
import secrets
import sys
import tomllib
import types
import typing
from collections.abc import Mapping
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

from .schema import Calibration, Model, Notify, Ntfy, SchemaError, ServerConfig, Tunnel, dump, load

ROOT = Path(__file__).resolve().parent.parent
STATE_DIR = ROOT / ".kis"
ENV_PREFIX = "KIS_"


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
    resident: list[str] = field(default_factory=list)
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


def with_env(cls: type, data: dict[str, Any], environ: Mapping[str, str], prefix: str = ENV_PREFIX) -> dict[str, Any]:
    """`data` (parsed TOML or JSON of dataclass `cls`) with the values set in the environment
    as KIS_ and the field's path in upper case, e.g. KIS_SERVER_ROLLOVER_HOURS for
    [server] rollover_hours. Strings are taken as they are, other values are read as TOML
    (2.5, true, ["--metrics"]). Presets ([models.*]) can't be set: their names may not be
    valid variable names."""
    data = dict(data)
    hints = typing.get_type_hints(cls)
    for f in dataclasses.fields(cls):
        name, hint = prefix + f.name.upper(), hints[f.name]
        if dataclasses.is_dataclass(hint):
            if table := with_env(hint, data.get(f.name, {}), environ, name + "_"):  # type: ignore[arg-type]
                data[f.name] = table
        elif name in environ and typing.get_origin(hint) is not dict:
            data[f.name] = environ[name] if _is_text(hint) else _toml_value(name, environ[name])
    return data


def _is_text(hint: Any) -> bool:
    """str, a str enum, or an optional one."""
    if typing.get_origin(hint) in (typing.Union, types.UnionType):
        return any(_is_text(a) for a in typing.get_args(hint))
    return isinstance(hint, type) and issubclass(hint, str)


def _toml_value(name: str, value: str) -> Any:
    try:
        return tomllib.loads(f"v = {value}")["v"]
    except tomllib.TOMLDecodeError as e:
        raise SchemaError(f"{name}: {value!r} is not a TOML value ({e})") from None


def parse_config(data: dict[str, Any], environ: Mapping[str, str] | None = None) -> Config:
    """Config from parsed TOML, with the values set in the environment (see with_env)."""
    return load(Config, with_env(Config, data, os.environ if environ is None else environ), "config")


def load_config(required: bool = True) -> Config:
    """config.toml, or the file named by KIS_CONFIG. Without one, unless `required`, the
    defaults and the environment: enough for `kis proxy`, which only follows the ntfy topic."""
    path = Path(os.environ.get("KIS_CONFIG") or ROOT / "config.toml")
    if not path.exists() and not required:
        return parse_config({"kaggle": {"username": "", "llama_cpp_ref": ""}, "models": {}})
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
    """API key and private ntfy topic from .kis/secrets.json and the environment (KIS_API_KEY,
    KIS_NTFY_TOPIC). The file is generated on first use, unless the environment sets both."""
    data = _read("secrets.json")
    if not data and len(with_env(Secrets, {}, os.environ)) < len(dataclasses.fields(Secrets)):
        data = {"api_key": "sk-kis-" + secrets.token_urlsafe(24), "ntfy_topic": "kis-" + secrets.token_hex(12)}
        _write("secrets.json", data)
    return load(Secrets, with_env(Secrets, data, os.environ), ".kis/secrets.json")


def load_state() -> State:
    """Saved state; keys of older kis versions are ignored, unreadable state starts over."""
    try:
        return load(State, _read("state.json"), ".kis/state.json", strict=False)
    except SchemaError as e:
        print(f"! ignoring {e}")
        return State()


def save_state(state: State) -> None:
    _write("state.json", dump(state))
