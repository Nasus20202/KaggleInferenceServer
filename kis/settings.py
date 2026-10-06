"""config.toml, generated secrets and local state (.kis/)."""

import json
import secrets
import sys
import tomllib
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
STATE_DIR = ROOT / ".kis"


def load_config() -> dict:
    path = ROOT / "config.toml"
    if not path.exists():
        sys.exit("config.toml missing: cp config.example.toml config.toml and set kaggle.username")
    config = tomllib.loads(path.read_text())
    if config["kaggle"]["username"] == "your-kaggle-username":
        sys.exit("set kaggle.username in config.toml")
    return config


def _read(name: str) -> dict:
    path = STATE_DIR / name
    return json.loads(path.read_text()) if path.exists() else {}


def _write(name: str, data: dict):
    STATE_DIR.mkdir(exist_ok=True)
    path = STATE_DIR / name
    path.write_text(json.dumps(data, indent=2))
    path.chmod(0o600)


def load_secrets() -> dict:
    """API key and private ntfy topic, generated on first use."""
    data = _read("secrets.json")
    if not data:
        data = {"api_key": "sk-kis-" + secrets.token_urlsafe(24), "ntfy_topic": "kis-" + secrets.token_hex(12)}
        _write("secrets.json", data)
    return data


def load_state() -> dict:
    return _read("state.json")


def save_state(**data):
    _write("state.json", {**load_state(), **data})
