"""Rendering kernel scripts from examples/config.32k.toml, and the model presets."""

import argparse
import re
import subprocess
import sys

import pytest

from kis import cli, kaggle

SECRETS = {"api_key": "secret", "ntfy_topic": "test"}


def up_args(**overrides) -> argparse.Namespace:
    return argparse.Namespace(
        **{"model": "qwen35-9b", "parallel": None, "topology": None, "ctx": None, "no_spec": False, **overrides}
    )


def rendered_config(config: dict, **overrides) -> dict:
    source = kaggle.render("server.py", cli.server_params(up_args(**overrides), config, SECRETS))
    compile(source, "server.py", "exec")
    namespace = {}
    exec(source[source.index("# <params>") : source.index("# </params>")], namespace)
    return namespace["CONFIG"]


def test_every_preset_renders(config):
    for name, preset in config["models"].items():
        rendered = rendered_config(config, model=name)
        assert rendered["model"] == preset
        assert rendered["api_key"] == "secret"
        assert rendered["args"] == config["server"]["args"]


def test_rendered_config_has_the_scripts_keys(config, server):
    assert rendered_config(config).keys() == server.CONFIG.keys()


def test_overrides(config):
    model = rendered_config(config, model="gemma-4-e4b", parallel=3, topology="split", no_spec=True)["model"]
    assert model["parallel"] == 3
    assert model["topology"] == "split"
    assert model["draft_file"] is None
    assert "--spec-type" not in model["args"] and "--spec-draft-n-max" not in model["args"]
    assert model["args"][:2] == ["--temp", "1.0"]


def test_drop_options():
    args = ["--spec-type", "draft-mtp", "--temp", "1.0", "--spec-draft-n-max", "2"]
    assert cli._drop_options(args, "--spec-type", "--spec-draft-n-max") == ["--temp", "1.0"]


def test_unknown_model_exits(config):
    with pytest.raises(SystemExit):
        cli.server_params(up_args(model="nope"), config, SECRETS)


def test_tailscale_needs_authkey(config):
    config["tunnel"]["kind"] = "tailscale"
    with pytest.raises(SystemExit):
        cli.server_params(up_args(), config, SECRETS)


def test_build_render_sets_ref():
    source = kaggle.render("build.py", {"LLAMA_CPP_REF": "b1234"})
    compile(source, "build.py", "exec")
    assert "LLAMA_CPP_REF = 'b1234'" in source


def test_presets_are_complete(config):
    for name, preset in config["models"].items():
        for key in ("repo", "revision", "file", "alias", "parallel"):
            assert preset.get(key), f"{name}: missing {key}"
        assert preset.get("topology", "replicas") in ("replicas", "split"), name
        for key in ("sha256", "draft_sha256"):
            if key in preset:
                assert re.fullmatch(r"[0-9a-f]{64}", preset[key]), f"{name}: bad {key}"
        if "draft_file" in preset:
            assert "draft_sha256" in preset, f"{name}: unpinned draft"


def test_kernel_state(monkeypatch, config):
    outputs = {
        "a": 'u/a has status "KernelWorkerStatus.RUNNING"',
        "b": "Cannot access kernel 'u/b' (Permission 'kernels.get' was denied).",
    }
    monkeypatch.setattr(kaggle, "status", lambda config, slug: outputs[slug])
    assert (kaggle.state(config, "a"), kaggle.state(config, "b")) == ("running", "-")


@pytest.mark.parametrize(
    ("script", "params"),
    [("server.py", lambda config: cli.server_params(up_args(), config, SECRETS)), ("calibrate.py", lambda config: {})],
)
def test_rendered_scripts_are_self_contained(script, params, config, tmp_path):
    """Kaggle runs one file: kaggle/common.py must be inlined, and the result must import
    without the repository on sys.path."""
    source = kaggle.render(script, params(config))
    assert "from common import" not in source and "def llama_command(" in source
    path = tmp_path / script
    path.write_text(source)
    probe = f"import runpy; runpy.run_path({str(path)!r}, run_name='probe')"
    result = subprocess.run([sys.executable, "-c", probe], cwd=tmp_path, capture_output=True, text=True)
    assert result.returncode == 0, result.stderr
