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

ROOT = pathlib.Path(__file__).resolve().parents[1]
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


def test_ctd_max_depth():
    ctd = read_ctd_cnv(str(CTD), params=moria05_params())
    assert ctd_depth(ctd).max() == pytest.approx(1073, abs=1.0)


def test_sync_correlation(sync):
    # golden W-sync correlation 0.984; our reference-W is a coarser mean -> >0.95
    assert sync.corr > 0.95


def test_maxdepth_matches_golden(sync):
    golden = load_p_struct(str(MAT))
    assert sync.maxdepth == pytest.approx(golden["maxdepth"], abs=1.0)
