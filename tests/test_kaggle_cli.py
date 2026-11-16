"""The kaggle CLI that pushes the Kaggle kernels must start from a lock-file install (the image)."""

import ast
import importlib.metadata
import subprocess
import sys
import tomllib
from pathlib import Path

from packaging.requirements import Requirement
from packaging.utils import canonicalize_name

from kis import kaggle


def _declared_closure() -> set[str]:
    """This project's dependencies and everything they pull in, in the installed environment."""
    pyproject = tomllib.loads((kaggle.ROOT / "pyproject.toml").read_text())
    todo = [Requirement(r).name for r in pyproject["project"]["dependencies"]]
    seen: set[str] = set()
    while todo:
        name = canonicalize_name(todo.pop())
        if name in seen:
            continue
        seen.add(name)
        for requirement in importlib.metadata.requires(name) or []:
            parsed = Requirement(requirement)
            if parsed.marker is None or parsed.marker.evaluate():
                todo.append(parsed.name)
    return seen


def _kaggle_third_party_modules() -> set[str]:
    """Top-level modules the installed kaggle package imports, other than the standard library."""
    package = Path(str(importlib.metadata.distribution("kaggle").locate_file("kaggle")))
    modules: set[str] = set()
    for path in package.rglob("*.py"):
        for node in ast.walk(ast.parse(path.read_text(encoding="utf-8"))):
            if isinstance(node, ast.Import):
                modules.update(alias.name.partition(".")[0] for alias in node.names)
            elif isinstance(node, ast.ImportFrom) and node.level == 0 and node.module:
                modules.add(node.module.partition(".")[0])
    return modules - set(sys.stdlib_module_names) - {"kaggle"}


def test_kaggle_imports_come_from_declared_dependencies():
    """kaggle imports modules it does not declare (six, in 2.2.x). A lock-file install then has no
    such module, and `kaggle kernels push` fails before it uploads anything."""
    declared = _declared_closure()
    owners = importlib.metadata.packages_distributions()
    undeclared = {
        module: owners.get(module, [])
        for module in sorted(_kaggle_third_party_modules())
        if not declared & {canonicalize_name(owner) for owner in owners.get(module, [])}
    }
    assert not undeclared, f"kaggle imports modules that no declared dependency provides: {undeclared}"


def test_kaggle_cli_starts():
    result = subprocess.run(
        [sys.executable, "-c", "import kaggle.cli"], capture_output=True, text=True, cwd=kaggle.ROOT
    )
    assert result.returncode == 0, result.stderr
