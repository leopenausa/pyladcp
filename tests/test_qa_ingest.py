"""Phase 0/1 — dual-head ingest validated against the MORIA-80 golden p-struct.

Confirms the PD0 reader + DualHead assembly reproduce the exact instrument config the
reference LDEO_IX run recorded (ensemble/bin counts, cell/blank, first-bin distance,
beam angle), and that the clean CTD .cnv reader returns a sane 1 Hz series.
"""

from __future__ import annotations

import pathlib

import numpy as np
import pytest

from ladcp.config import moria05_params
from ladcp.io.ctd_cnv import read_ctd_cnv
from ladcp.qa.golden import load_p_struct
from ladcp.qa.ingest import extract_config, load_dualhead

ROOT = pathlib.Path(__file__).resolve().parents[1]
GOOD = ROOT / "New_golden" / "Good"
DOWN = GOOD / "LADCP" / "MORIA-80-LADCP-M.000"
UP = GOOD / "LADCP" / "MORIA-80-LADCP-S.000"
CTD = GOOD / "CTD" / "moria-80_clean.cnv"
MAT = ROOT / "New_golden" / "MORIA-80_LEO" / "MORIA-80.mat"

pytestmark = pytest.mark.skipif(not DOWN.exists(), reason="MORIA New_golden not present")


@pytest.fixture(scope="module")
def dh():
    return load_dualhead(str(DOWN), str(UP), station="MORIA-80")


@pytest.fixture(scope="module")
def golden():
    return load_p_struct(str(MAT))


def test_ensemble_counts(dh, golden):
    assert [dh.down.n_ens, dh.up.n_ens] == list(golden["nping_total"])


def test_bin_and_cell_config(dh, golden):
    cfg = extract_config(dh)
    assert cfg["nbin_d"] == golden["nbin_d"]
    assert cfg["nbin_u"] == golden["nbin_u"]
    assert cfg["blen_d"] == pytest.approx(golden["blen_d"])
    assert cfg["blnk_d"] == pytest.approx(golden["blnk_d"])
    assert cfg["beamangle"] == golden["beamangle"]


def test_first_bin_distance(dh, golden):
    cfg = extract_config(dh)
    assert cfg["dist_d"] == pytest.approx(golden["dist_d"], abs=0.02)
    assert cfg["dist_u"] == pytest.approx(golden["dist_u"], abs=0.02)


def test_bin_depth_axis(dh, golden):
    z = dh.bin_depth(dh.down)
    assert z[0] == pytest.approx(golden["dist_d"], abs=0.02)
    assert z[1] - z[0] == pytest.approx(dh.down.cell_m)
    assert z.size == dh.down.n_cells


def test_coord_frame_earth(dh):
    # MORIA deployed with EX11111 -> earth coordinates
    assert dh.down.coord_frame.value == "earth"
    assert dh.up.coord_frame.value == "earth"


def test_ctd_reader(dh):
    ctd = read_ctd_cnv(str(CTD), params=moria05_params())
    assert ctd.n > 3000
    dt = np.median(np.diff(ctd.time_elapsed_s))
    assert dt == pytest.approx(1.0, abs=0.05)          # 1 s bin average
    assert np.nanmax(ctd.pressure) > 900               # ~1073 dbar cast
    assert 30 < np.nanmedian(ctd.salinity) < 36
