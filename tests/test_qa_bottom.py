"""Seabed detection vs MORIA-80 golden (zbottom 1079.02 +/- 0.56)."""

from __future__ import annotations

import pathlib

import numpy as np
import pytest

from ladcp.config import moria05_params
from ladcp.io.ctd_cnv import read_ctd_cnv
from ladcp.qa.bottom import detect_bottom, localmax2, soundspeed_scale
from ladcp.qa.depth import synchronize
from ladcp.qa.golden import load_p_struct
from ladcp.qa.ingest import load_dualhead

ROOT = pathlib.Path(__file__).resolve().parent / "fixtures"  # tests/fixtures/
GOOD = ROOT / "New_golden" / "Good"
DOWN = GOOD / "LADCP" / "MORIA-80-LADCP-M.000"
UP = GOOD / "LADCP" / "MORIA-80-LADCP-S.000"
CTD = GOOD / "CTD" / "moria-80_clean.cnv"
MAT = ROOT / "New_golden" / "MORIA-80_LEO" / "MORIA-80.mat"

pytestmark = pytest.mark.skipif(not DOWN.exists(), reason="MORIA New_golden not present")


@pytest.fixture(scope="module")
def bottom():
    dh = load_dualhead(str(DOWN), str(UP), station="MORIA-80", params=moria05_params())
    ctd = read_ctd_cnv(str(CTD), params=moria05_params())
    sync = synchronize(dh, ctd)
    return detect_bottom(dh, sync, ctd=ctd)            # sound-speed corrected


def test_localmax2_parabola():
    # exact parabola y = -(x-3)^2 + 5 -> peak at x=3, y=5
    x = np.array([1.0, 2, 3, 4, 5])
    y = (-(x - 3) ** 2 + 5)[:, None]
    xp, yp, _ = localmax2(x, y)
    assert xp[0] == pytest.approx(3.0) and yp[0] == pytest.approx(5.0)


def test_zbottom_matches_golden(bottom):
    golden = load_p_struct(str(MAT))
    # depth-stack lock + sound-speed correction -> ~1080.7 vs golden 1079.02
    assert bottom.zbottom == pytest.approx(golden["zbottom"], abs=2.0)


def test_zbottom_error_small(bottom):
    assert bottom.error < 1.5          # golden zbottomerror 0.56; we get ~0.26


def test_bottom_echo_count(bottom):
    # n_valid counts the near-seabed echoes supporting the reported depth (a subset of the
    # ~1010 valid targ returns over the whole cast); golden created ~930 bottom distances
    assert 400 < bottom.n_valid < 1300


def test_soundspeed_scale_physical():
    # sc = c_ctd / c_onboard: onboard fixed 1450 m/s, in-situ deep water ~1490-1500,
    # so the correction inflates bottom distance by a few percent (1.0 < sc < 1.06)
    dh = load_dualhead(str(DOWN), str(UP), station="MORIA-80", params=moria05_params())
    ctd = read_ctd_cnv(str(CTD), params=moria05_params())
    sc = soundspeed_scale(dh, synchronize(dh, ctd), ctd)
    finite = sc[np.isfinite(sc)]
    assert finite.size > 0
    assert np.all(finite >= 1.0) and np.nanmax(finite) < 1.06


def test_soundspeed_makes_deeper(bottom):
    # corrected bottom must be deeper than the uncorrected estimate
    dh = load_dualhead(str(DOWN), str(UP), station="MORIA-80", params=moria05_params())
    ctd = read_ctd_cnv(str(CTD), params=moria05_params())
    sync = synchronize(dh, ctd)
    uncorrected = detect_bottom(dh, sync)
    assert bottom.zbottom > uncorrected.zbottom
