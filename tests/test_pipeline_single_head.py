"""The pipeline's single-head gate: a cast with no up-looker solves automatically.

A synthetic station (down-looker only handed to ``process_station``) must produce a
velocity solution without ``--down-only``, carrying the ``single_head_solve`` WARN
metric that says the down-only solve was automatic. With the flag set explicitly on
a dual-head cast the same metric appears without the "automatic" wording.
"""

from __future__ import annotations

import json

import matplotlib
import pytest

from ladcp.qa.pipeline import process_station
from ladcp.session import EditConfig, SessionConfig
from ladcp.synth import SynthConfig, generate_station

matplotlib.use("Agg")


@pytest.fixture(scope="module")
def synth(tmp_path_factory):
    root = tmp_path_factory.mktemp("synth_raw")
    cfg = SynthConfig(out=root, station="SYNTH-01", seed=0, noise=0.0)
    paths, _ = generate_station(cfg)
    return paths


def _run(paths, out, up, cfg):
    return process_station(str(paths.down), up, str(paths.ctd), "SYNTH-01",
                           str(out), False, cfg, cruise="LADCP", formats=None)


def test_no_up_looker_solves_automatically(synth, tmp_path):
    _, export = _run(synth, tmp_path, None, SessionConfig())
    st = tmp_path / "stations" / "SYNTH-01"
    assert (st / "SYNTH-01.lad").is_file()               # a solution was produced
    assert export is not None and export.result is not None
    m = json.loads((st / "SYNTH-01_qa.json").read_text())["metrics"]["single_head_solve"]
    assert m["status"] == "warn" and "automatic" in m["note"]


def test_explicit_down_only_still_excludes_up(synth, tmp_path):
    _, export = _run(synth, tmp_path, str(synth.up),
                     SessionConfig(edit=EditConfig(down_only=True)))
    st = tmp_path / "stations" / "SYNTH-01"
    assert (st / "SYNTH-01.lad").is_file()
    m = json.loads((st / "SYNTH-01_qa.json").read_text())["metrics"]["single_head_solve"]
    assert m["status"] == "warn" and "automatic" not in m["note"]


def test_dual_head_has_no_single_head_metric(synth, tmp_path):
    _, _ = _run(synth, tmp_path, str(synth.up), SessionConfig())
    st = tmp_path / "stations" / "SYNTH-01"
    assert (st / "SYNTH-01.lad").is_file()
    metrics = json.loads((st / "SYNTH-01_qa.json").read_text())["metrics"]
    assert "single_head_solve" not in metrics
    assert "velocity_skipped" not in metrics
