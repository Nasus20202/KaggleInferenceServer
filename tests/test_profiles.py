"""examples/config.<profile>.toml and the README table from calibration results."""

import tomllib

from kis import profiles


def fit(parallel: int, ctx: int, topology: str = "replicas") -> dict:
    instances = 2 if topology == "replicas" else 1
    return {"parallel": parallel, "ctx": ctx, "topology": topology, "instances": instances, "single_tps": 50.0}


RESULTS = {
    "qwen35-4b": {"profiles": {"128k": fit(2, 131072), "max": fit(1, 262144, "split")}},
    "qwen38-27b": {"profiles": {"max": fit(1, 196608, "split")}},
}


def test_sets_slots_context_and_topology_and_skips_what_does_not_fit():
    template = profiles.TEMPLATE.read_text()
    config = tomllib.loads(profiles.render(template, "max", RESULTS))
    assert {"qwen35-4b", "qwen38-27b", "gemma-4-e2b"} <= set(config["models"])  # unmeasured presets stay
    small = config["models"]["qwen35-4b"]
    assert (small["parallel"], small["ctx"], small["topology"]) == (1, 262144, "split")
    big = config["models"]["qwen38-27b"]
    assert (big["ctx"], big["topology"], big["tensor_split"]) == (196608, "split", "5,4")

    config = tomllib.loads(profiles.render(template, "128k", RESULTS))
    assert "qwen38-27b" not in config["models"]  # measured: doesn't fit 128K
    assert "topology" not in config["models"]["qwen35-4b"]


def test_render_is_stable_when_rerun_on_its_output():
    once = profiles.render(profiles.TEMPLATE.read_text(), "max", RESULTS)
    assert profiles.render(once, "max", RESULTS) == once


def test_table_shows_slots_context_and_speed():
    table = profiles.table(RESULTS)
    assert "| `qwen35-4b` | - | - | - | 4 x 128K<br>50.0 / ? tok/s | 1 x 256K split<br>50.0 / ? tok/s |" in table
