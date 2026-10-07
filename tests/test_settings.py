"""config.toml, secrets and their environment overrides, as kis/settings.py loads them."""

import pytest

from kis import settings
from kis.schema import SchemaError, TunnelKind
from kis.settings import Secrets, parse_config


def test_every_config_value_can_be_set_in_the_environment(config):
    data = {"kaggle": {"username": "u", "llama_cpp_ref": "v1"}, "models": {}}
    environ = {
        "KIS_KAGGLE_USERNAME": "123",  # a string field stays a string
        "KIS_SERVER_ROLLOVER_HOURS": "2.5",
        "KIS_SERVER_AUTOLOAD": "false",
        "KIS_SERVER_ARGS": '["--metrics"]',
        "KIS_TUNNEL_KIND": "tailscale",
        "KIS_NTFY_TOKEN": "tk_x",
    }
    config = parse_config(data, environ)
    assert config.kaggle.username == "123"
    assert (config.server.rollover_hours, config.server.autoload, config.server.args) == (2.5, False, ["--metrics"])
    assert (config.tunnel.kind, config.ntfy.token) == (TunnelKind.TAILSCALE, "tk_x")
    with pytest.raises(SchemaError, match="KIS_SERVER_CTX"):
        parse_config(data, {"KIS_SERVER_CTX": "lots"})


def test_secrets_from_the_environment_need_no_file(monkeypatch, tmp_path):
    monkeypatch.setattr(settings, "STATE_DIR", tmp_path)
    monkeypatch.setenv("KIS_API_KEY", "sk-env")
    monkeypatch.setenv("KIS_NTFY_TOPIC", "kis-env")
    assert settings.load_secrets() == Secrets(api_key="sk-env", ntfy_topic="kis-env")
    assert not (tmp_path / "secrets.json").exists()
    monkeypatch.delenv("KIS_NTFY_TOPIC")
    secrets = settings.load_secrets()  # generated on first use; the environment still overrides
    assert secrets.api_key == "sk-env" and secrets.ntfy_topic.startswith("kis-")
    assert (tmp_path / "secrets.json").exists()


def test_only_the_proxy_runs_without_a_config(monkeypatch, tmp_path):
    monkeypatch.setenv("KIS_CONFIG", str(tmp_path / "missing.toml"))
    monkeypatch.setenv("KIS_NTFY_SERVER", "https://ntfy.example.com")
    config = settings.load_config(required=False)
    assert config.models == {} and config.server.rollover_hours == 0
    assert config.ntfy.server == "https://ntfy.example.com"
    with pytest.raises(SystemExit):
        settings.load_config()
