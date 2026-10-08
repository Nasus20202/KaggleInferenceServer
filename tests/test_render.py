"""Kaggle runs one file: the rendered scripts must be one consistent module namespace.

`kis.kaggle.render` pastes every INLINED module over its import line, so all of them share
the namespace of the script. These tests render each script and check what splitting
kaggle/ into modules could break: an import the paster misses, a name defined twice, a
module pasted after the code that needs it.
"""

import argparse
import ast
from collections import defaultdict

import pytest
from conftest import SECRETS

from kis import cli, kaggle
from kis.schema import CalibrateConfig

SCRIPTS = {
    "server.py": lambda config: {
        "CONFIG": cli.server_config(
            argparse.Namespace(model="qwen35-9b", parallel=None, topology=None, ctx=None, no_spec=False),
            config,
            SECRETS,
        )
    },
    "calibrate.py": lambda config: {"CONFIG": CalibrateConfig(run="1", models=config.models)},
    "build.py": lambda config: {"LLAMA_CPP_REF": "b1234"},
}


@pytest.fixture(params=SCRIPTS)
def rendered(request, config) -> tuple[str, str]:
    script = request.param
    return script, kaggle.render(script, SCRIPTS[script](config))


def bindings(tree: ast.Module) -> dict[str, list[str]]:
    """Top-level name -> where each binding comes from: `import module.name`, or `def@line`."""
    found: dict[str, list[str]] = defaultdict(list)

    def bind(name: str, origin: str) -> None:
        found[name].append(origin)

    for node in tree.body:
        match node:
            case ast.FunctionDef(name=name) | ast.AsyncFunctionDef(name=name) | ast.ClassDef(name=name):
                bind(name, f"def@{node.lineno}")
            case ast.Assign(targets=targets):
                for target in targets:
                    for name in ast.walk(target):
                        if isinstance(name, ast.Name):
                            bind(name.id, f"def@{node.lineno}")
            case ast.AnnAssign(target=ast.Name(id=name)):
                bind(name, f"def@{node.lineno}")
            case ast.Import():
                for alias in node.names:
                    # `import a.b` binds `a`, whatever the submodule
                    bind(
                        (alias.asname or alias.name).split(".")[0], f"import {alias.asname or alias.name.split('.')[0]}"
                    )
            case ast.ImportFrom(module=module):
                for alias in node.names:
                    bind(alias.asname or alias.name, f"import {module}.{alias.name}")
    return found


def test_rendered_script_compiles(rendered):
    script, source = rendered
    compile(source, script, "exec")


def test_no_top_level_name_is_defined_twice(rendered):
    """Every module ends up in one namespace: a second `def`, class or assignment of a name
    silently replaces the first, and one name imported from two places is a clash too.
    The same import twice is harmless."""
    script, source = rendered
    clashes = {}
    for name, origins in bindings(ast.parse(source)).items():
        if len(set(origins)) > 1 or sum(o.startswith("def@") for o in origins) > 1:
            clashes[name] = origins
    assert not clashes, f"{script}: {clashes}"


def test_no_import_of_an_inlined_module_is_left(rendered):
    """The paster only handles `from <module> import a, b` (or a parenthesized list) at the
    start of a line; any other way to import an INLINED module would fail on Kaggle."""
    script, source = rendered
    inlined = set(kaggle.INLINED)
    left = []
    for node in ast.walk(ast.parse(source)):
        if isinstance(node, ast.Import):
            left += [a.name for a in node.names if a.name in inlined or a.name.split(".")[0] in inlined]
        elif isinstance(node, ast.ImportFrom):
            if node.level:
                left.append("." * node.level + (node.module or ""))
            elif node.module in inlined or (node.module or "").split(".")[0] in inlined:
                left.append(node.module or "")
    assert not left, f"{script}: imports that were not pasted: {left}"
    stars = [n.module for n in ast.walk(ast.parse(source)) if isinstance(n, ast.ImportFrom) and n.names[0].name == "*"]
    assert not stars, f"{script}: star imports hide name clashes: {stars}"


def test_each_module_is_pasted_once(rendered):
    script, source = rendered
    markers = [line for line in source.splitlines() if line.startswith("# --- ")]
    assert len(markers) == len(set(markers)), f"{script}: a module is pasted twice: {markers}"


def test_rendered_top_level_runs(rendered):
    """Execute the top level (definitions, constants, settings) in a throwaway namespace: a
    module pasted after the code that needs it fails here with a NameError. build.py runs
    cmake as soon as it is executed, so it is only compiled."""
    script, source = rendered
    if script == "build.py":
        pytest.skip("build.py has no definitions to check; it compiles llama.cpp when run")
    namespace = {"__name__": "probe", "__file__": script}
    exec(compile(source, script, "exec"), namespace)
    assert "SETTINGS" in namespace
