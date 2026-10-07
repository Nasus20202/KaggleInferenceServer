"""Rendering kernel scripts from examples/config.32k.toml, and the model presets."""

import argparse
import dataclasses
import re
import subprocess
import sys
import tomllib

import pytest
from conftest import SECRETS

from kis import cli, kaggle
from kis.schema import CalibrateConfig, Model, Ntfy, SchemaError, ServerConfig, load
from kis.settings import parse_config


def up_args(**overrides) -> argparse.Namespace:
    return argparse.Namespace(
        **{"model": "qwen35-9b", "parallel": None, "topology": None, "ctx": None, "no_spec": False, **overrides}
    )


def rendered_config(config, **overrides) -> dict:
    source = kaggle.render("server.py", {"CONFIG": cli.server_config(up_args(**overrides), config, SECRETS)})
    compile(source, "server.py", "exec")
    namespace = {}
    exec(source[source.index("# <params>") : source.index("# </params>")], namespace)
    return namespace["CONFIG"]


def test_every_preset_renders(config):
    for name, preset in config.models.items():
        rendered = rendered_config(config, model=name)
        assert load(Model, rendered["model"]) == preset
        assert rendered["api_key"] == "secret"
        assert rendered["args"] == config.server.args


def test_rendered_config_has_the_scripts_keys(config, server):
    assert rendered_config(config).keys() == server.CONFIG.keys()
    assert load(ServerConfig, rendered_config(config)).model.alias == config.models["qwen35-9b"].alias


def test_overrides(config):
    model = load(
        Model, rendered_config(config, model="gemma-4-e4b", parallel=3, topology="split", no_spec=True)["model"]
    )
    assert model.parallel == 3
    assert model.topology == "split"
    assert model.draft_file is None
    assert "--spec-type" not in model.args and "--spec-draft-n-max" not in model.args
    assert model.args[:2] == ["--temp", "1.0"]
    assert config.models["gemma-4-e4b"].parallel != 3  # the preset itself is unchanged


def test_drop_options():
    args = ["--spec-type", "draft-mtp", "--temp", "1.0", "--spec-draft-n-max", "2"]
    assert cli._drop_options(args, "--spec-type", "--spec-draft-n-max") == ["--temp", "1.0"]


def test_unknown_model_exits(config):
    with pytest.raises(SystemExit):
        cli.server_config(up_args(model="nope"), config, SECRETS)


def test_tailscale_needs_authkey(config):
    config.tunnel = dataclasses.replace(config.tunnel, kind="tailscale")
    with pytest.raises(SystemExit):
        cli.server_config(up_args(), config, SECRETS)


def test_build_render_sets_ref():
    source = kaggle.render("build.py", {"LLAMA_CPP_REF": "b1234"})
    compile(source, "build.py", "exec")
    assert "LLAMA_CPP_REF = 'b1234'" in source


def test_presets_are_complete(config):
    for name, preset in config.models.items():
        assert preset.revision, f"{name}: unpinned revision"
        for digest in (preset.sha256, preset.draft_sha256):
            if digest is not None:
                assert re.fullmatch(r"[0-9a-f]{64}", digest), f"{name}: bad sha256"
        if preset.draft_file:
            assert preset.draft_sha256, f"{name}: unpinned draft"


def test_every_example_config_loads():
    for path in sorted(kaggle.ROOT.glob("examples/config.*.toml")):
        config = parse_config(tomllib.loads(path.read_text()), environ={})
        assert config.models, path


def test_config_errors_name_the_key(config):
    data = tomllib.loads((kaggle.ROOT / "examples" / "config.32k.toml").read_text())
    data["models"]["qwen35-9b"]["parallel"] = "4"
    with pytest.raises(SchemaError, match=r"config\.models\.qwen35-9b\.parallel: expected int"):
        parse_config(data, environ={})
    data["models"]["qwen35-9b"]["parallel"] = 4
    data["server"]["idle_minuts"] = 5
    with pytest.raises(SchemaError, match=r"config\.server: unknown key"):
        parse_config(data, environ={})
    del data["server"]["idle_minuts"]
    data["tunnel"]["kind"] = "ngrok"
    with pytest.raises(SchemaError, match="not one of 'cloudflared', 'tailscale'"):
        parse_config(data, environ={})


def test_ntfy_server_and_tokens(config):
    data = tomllib.loads((kaggle.ROOT / "examples" / "config.32k.toml").read_text())
    data["ntfy"] = {"server": "https://ntfy.example.com"}
    data["notify"] = {"topic": "phone"}
    config = parse_config(data, environ={"KIS_NTFY_TOKEN": "tk_control"})
    assert config.ntfy == Ntfy("https://ntfy.example.com", "tk_control")
    assert config.notify.via(config.ntfy) == Ntfy("https://ntfy.example.com", "tk_control")  # same server
    other = dataclasses.replace(config.notify, server="https://ntfy.sh")
    assert other.via(config.ntfy) == Ntfy("https://ntfy.sh", "")  # the control token stays on its server
    rendered = rendered_config(config)
    assert rendered["ntfy"] == {"server": "https://ntfy.example.com", "token": "tk_control"}


def test_kernel_state(monkeypatch, config):
    outputs = {
        "a": 'u/a has status "KernelWorkerStatus.RUNNING"',
        "b": "Cannot access kernel 'u/b' (Permission 'kernels.get' was denied).",
    }
    monkeypatch.setattr(kaggle, "status", lambda config, slug: outputs[slug])
    assert (kaggle.state(config, "a"), kaggle.state(config, "b")) == ("running", "-")


@pytest.mark.parametrize(
    ("script", "params"),
    [
        ("server.py", lambda config: {"CONFIG": cli.server_config(up_args(), config, SECRETS)}),
        ("calibrate.py", lambda config: {"CONFIG": CalibrateConfig(run="1", models=config.models)}),
    ],
)
def test_rendered_scripts_are_self_contained(script, params, config, tmp_path):
    """Kaggle runs one file: kaggle/common.py and kis/schema.py must be inlined (once), and
    the result must import without the repository on sys.path."""
    source = kaggle.render(script, params(config))
    assert "from common import" not in source and "def llama_command(" in source
    assert "from kis" not in source and source.count("class ServerConfig") == 1
    path = tmp_path / script
    path.write_text(source)
    probe = f"import runpy; runpy.run_path({str(path)!r}, run_name='probe')"
    result = subprocess.run([sys.executable, "-c", probe], cwd=tmp_path, capture_output=True, text=True)
    assert result.returncode == 0, result.stderr
