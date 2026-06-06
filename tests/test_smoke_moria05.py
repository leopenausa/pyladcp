"""Smoke tests: readers + harness run on the MORIA-05 fixtures present in the repo."""

from pathlib import Path

import numpy as np
import pytest

from ladcp.config import moria05_params
from ladcp.io.ctd import read_clean_ctd
from ladcp.io.golden import read_lad, scan_log
from ladcp.io.pd0 import read_pd0
from ladcp.validate.harness import moria05_inputs, run

ROOT = Path(__file__).resolve().parents[1]
HAVE_DATA = (ROOT / "raw_ladcp_test/LADCP/Data/MASTER/MLADC007.000").exists()
needs_data = pytest.mark.skipif(not HAVE_DATA, reason="MORIA raw fixtures not present")


@needs_data
def test_pd0_master_config():
    m = read_pd0(str(ROOT / "raw_ladcp_test/LADCP/Data/MASTER/MLADC007.000"),
                 head="downlooker", facing_hint="down")
    assert m.freq_khz == 300
    assert m.n_cells == 30
    assert abs(m.cell_m - 8.0) < 1e-6
    assert m.facing == "down"
    assert m.coord_frame.value == "earth"
    assert m.bt_vel is not None and np.isfinite(m.bt_vel).any()


@needs_data
def test_ctd_columns():
    ctd = read_clean_ctd(str(ROOT / "clean_ctd/moria-05_clean.cnv"), moria05_params())
    assert ctd.n > 1000
    assert 46.0 < ctd.lat[0] < 47.0
    assert -6.5 < ctd.lon[0] < -5.0
    assert np.nanmax(ctd.pressure) > 4000


@needs_data
def test_golden_lad_and_log():
    g = read_lad(str(ROOT / "figures/MORIA-05.lad"))
    assert g.z[0] == 8.0 and g.z.size > 500
    assert abs(g.header["deviation"] - (-2.545590)) < 1e-4
    log = scan_log(str(ROOT / "figures/MORIA-05.log"))
    assert log["sadcp_used"] is False


@needs_data
def test_harness_runs_and_config_gates_pass():
    rep = run(moria05_inputs(ROOT))
    by = {c.name: c for c in rep.checks}
    for gate in ("master_is_downlooker", "slave_is_uplooker", "freq_khz",
                 "n_cells", "cell_size_m", "bottom_track_present"):
        assert by[gate].status == "pass", gate
    # inversion runs end-to-end, detects the bottom, and reproduces the golden
    # absolute-velocity profile within the accepted 1 cm/s tolerance.
    assert by["inversion_ran"].status == "pass"
    assert by["bottom_depth_m"].status == "pass"
    assert by["u_profile"].status == "pass", by["u_profile"].detail
    assert by["v_profile"].status == "pass", by["v_profile"].detail
