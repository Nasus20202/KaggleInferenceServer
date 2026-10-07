"""Example configs and the README table, generated from `kis calibrate` results.

examples/config.<profile>.toml share the presets of examples/config.32k.toml and
differ in context per slot, slots and, where a profile doesn't fit one GPU, topology.
"""

import re

from .schema import MAX_PROFILE, PROFILE_CTX, Calibration, Topology
from .settings import ROOT

PROFILES = {
    **{name: f"{ctx // 1024}K tokens of context per slot, as many slots as fit" for name, ctx in PROFILE_CTX.items()},
    MAX_PROFILE: "the largest context per slot that fits (up to the model's trained context), "
    "then as many slots as fit",
}
EXAMPLES = ROOT / "examples"
TEMPLATE = EXAMPLES / "config.32k.toml"
PRESETS_HEADER = re.compile(r"# -{70,}\n# Model presets.*?# -{70,}", re.S)
README_TABLE = re.compile(r"(<!-- calibration -->\n).*?(<!-- /calibration -->)", re.S)


def render(template: str, profile: str, results: dict[str, Calibration]) -> str:
    """The template config with each preset's slots, context and topology set from `results`.
    Presets measured not to fit the profile are left out; unmeasured ones stay unchanged."""
    header = (
        "# " + "-" * 75 + "\n"
        f"# Model presets, profile {profile}: {PROFILES[profile]}.\n"
        "# Measured on Kaggle's 2x T4 with `kis calibrate`; `kis calibrate --write` regenerates\n"
        "# this file. `parallel` = slots per llama-server instance (2 replicas -> 2x slots),\n"
        "# `ctx` = context each slot is guaranteed. KV cache in f16 (no quality loss).\n"
        "# `alias` is the model name the API reports. Sampling follows each model card.\n"
        "# " + "-" * 75
    )
    text = PRESETS_HEADER.sub(lambda _: header, template, count=1)
    out: list[str] = []
    skip, fit = False, None
    for line in text.splitlines(keepends=True):
        if m := re.match(r"\[models\.([\w.-]+)\]", line):
            model = m.group(1)
            fit = results[model].profiles.get(profile) if model in results else None
            skip = model in results and not fit
        if skip:
            continue
        if fit and re.match(r"(ctx|topology) = ", line):
            continue
        if fit and re.match(r"parallel = ", line):
            line = f"parallel = {fit.parallel}\nctx = {fit.ctx}\n"
            if fit.topology == Topology.SPLIT:
                line = f'topology = "{Topology.SPLIT}"\n' + line
        out.append(line)
    return "".join(out)


def table(results: dict[str, Calibration]) -> str:
    """Markdown: slots x context and tok/s (one request / all slots busy) per model and profile."""
    head = "| model | " + " | ".join(PROFILES) + " |\n| --- |" + " --- |" * len(PROFILES) + "\n"
    rows = []
    for name, r in results.items():
        cells = []
        for profile in PROFILES:
            p = r.profiles.get(profile)
            if not p:
                cells.append("-")
                continue
            split = " split" if p.topology == Topology.SPLIT else ""
            speed = f"<br>{p.single_tps or '?'} / {p.total_tps or '?'} tok/s"
            cells.append(f"{p.slots} x {p.ctx // 1024}K{split}{speed}")
        rows.append(f"| `{name}` | " + " | ".join(cells) + " |")
    return head + "\n".join(rows) + "\n"


def write(results: dict[str, Calibration]) -> None:
    template = TEMPLATE.read_text()
    for profile in PROFILES:
        (EXAMPLES / f"config.{profile}.toml").write_text(render(template, profile, results))
    readme = ROOT / "README.md"
    text, n = README_TABLE.subn(lambda m: m.group(1) + table(results) + m.group(2), readme.read_text())
    if n:
        readme.write_text(text)
