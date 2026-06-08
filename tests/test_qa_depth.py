"""CTD<->LADCP synchronization + package depth vs MORIA-80 golden."""

from __future__ import annotations

import pathlib

import pytest

from ladcp.config import moria05_params
from ladcp.io.ctd_cnv import read_ctd_cnv
from ladcp.qa.bestlag import bestlag
from ladcp.qa.depth import ctd_depth, synchronize
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
def sync():
    dh = load_dualhead(str(DOWN), str(UP), station="MORIA-80", params=moria05_params())
    ctd = read_ctd_cnv(str(CTD), params=moria05_params())
    return synchronize(dh, ctd)


def test_bestlag_identity():
    import numpy as np
    x = np.sin(np.linspace(0, 40, 500))
    lag, co = bestlag(x, np.roll(x, 7), nlag=30)
    assert lag == 7 and co > 0.99


def test_coarse_offset_locates_buried_cast():
    # the ADCP commonly pings on deck long before/after the cast, so the cast's vertical-velocity
    # signature sits thousands of pings into the record -- a clock offset beyond bestlag's reach.
    # _coarse_offset must locate it by sliding the CTD W window across the whole ADCP series.
    import numpy as np

    from ladcp.qa.depth import _coarse_offset

    t_ctd = np.arange(0.0, 200.0)
    w_ctd = np.concatenate([np.full(100, 1.0), np.full(100, -1.0)])   # descent then ascent
    tad = np.arange(0.0, 2000.0)
    w_ad = np.zeros(2000)
    w_ad[1500:1700] = w_ctd                                            # bump buried at t=1500 s
    offset, score = _coarse_offset(tad, w_ad, t_ctd, w_ctd)
    assert abs(offset - 1500.0) <= 1.0      # CTD t=0 maps to ADCP t=1500
    assert score > 0.9
    # too-short / empty inputs degrade gracefully to (0, 0), never raising
    assert _coarse_offset(np.array([0.0]), np.array([0.0]), t_ctd, w_ctd) == (0.0, 0.0)


def test_sync_offset_handles_large_clock_skew(sync):
    # the MORIA-80 golden offset (~431 s) is well within the old +-600 single-stage reach, so the
    # coarse+fine pipeline must reproduce it (no regression) and report a real coarse score.
    assert abs(sync.lag - 431) <= 2
    assert sync.coarse_score > 0.5


def test_ctd_max_depth():
    ctd = read_ctd_cnv(str(CTD), params=moria05_params())
    assert ctd_depth(ctd).max() == pytest.approx(1073, abs=1.0)


def test_sync_correlation(sync):
    # golden W-sync correlation 0.984; our reference-W is a coarser mean -> >0.95
    assert sync.corr > 0.95


def test_maxdepth_matches_golden(sync):
    golden = load_p_struct(str(MAT))
    assert sync.maxdepth == pytest.approx(golden["maxdepth"], abs=1.0)
