"""Surface/seabed detection: water-window + Fig 4 render."""

from __future__ import annotations

import pathlib

import matplotlib
import numpy as np
import pytest

matplotlib.use("Agg")

from ladcp.config import moria05_params
from ladcp.io.ctd_cnv import read_ctd_cnv
from ladcp.plots.depth_figure import depth_figure
from ladcp.qa.depth import synchronize, water_window
from ladcp.qa.ingest import load_dualhead

ROOT = pathlib.Path(__file__).resolve().parent / "fixtures"  # tests/fixtures/
GOOD = ROOT / "New_golden" / "Good"
DOWN = GOOD / "LADCP" / "MORIA-80-LADCP-M.000"
UP = GOOD / "LADCP" / "MORIA-80-LADCP-S.000"
CTD = GOOD / "CTD" / "moria-80_clean.cnv"

pytestmark = pytest.mark.skipif(not DOWN.exists(), reason="MORIA New_golden not present")


def test_water_window_spans_cast():
    dh = load_dualhead(str(DOWN), str(UP), params=moria05_params())
    ctd = read_ctd_cnv(str(CTD), params=moria05_params())
    sync = synchronize(dh, ctd)
    i0, i1 = water_window(sync.z_on_ping)
    assert 0 < i0 < i1 < dh.down.n_ens
    # deepest ping must lie inside the in-water window
    assert i0 < int(np.nanargmax(sync.z_on_ping)) < i1


def test_depth_figure_renders(tmp_path):
    dh = load_dualhead(str(DOWN), str(UP), station="MORIA-80", params=moria05_params())
    ctd = read_ctd_cnv(str(CTD), params=moria05_params())
    out = tmp_path / "depth.png"
    depth_figure(dh, ctd, savepath=str(out))
    assert out.exists() and out.stat().st_size > 0
