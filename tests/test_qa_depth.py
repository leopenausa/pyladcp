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


# --- water_window: deep-segment semantics (CRUISE2 t99-03 / t1-01) --------------------
def test_water_window_normal_cast_unchanged():
    import numpy as np

    from ladcp.qa.depth import water_window
    z = np.concatenate([np.linspace(0, 500, 100), np.linspace(500, 0, 100)])
    i0, i1 = water_window(z)
    assert z[i0 - 1] <= 10.0 < z[i0] and z[i1 + 1] <= 10.0 < z[i1]


def test_water_window_excludes_pre_cast_soak():
    import numpy as np

    from ladcp.qa.depth import water_window
    # 200-ping soak at 12-15 m, a 90-ping near-surface gap, then the real cast
    soak = np.full(200, 13.0)
    gap = np.full(90, 6.0)
    cast = np.concatenate([np.linspace(6, 400, 150), np.linspace(400, 6, 150)])
    z = np.concatenate([soak, gap, cast])
    i0, i1 = water_window(z)
    assert i0 >= 290                                  # the soak + gap are excluded
    assert z[i0] > 10.0


def test_water_window_bridges_threshold_flicker():
    import numpy as np

    from ladcp.qa.depth import water_window
    # the package hangs right at the threshold near the end (t1-01): single-ping
    # dips to 10.0 must not cut the window short
    cast = np.concatenate([np.linspace(8, 120, 80), np.linspace(120, 11, 80)])
    hover = np.full(120, 10.4)
    hover[::7] = 10.0                                 # flicker below the threshold
    z = np.concatenate([cast, hover])
    i0, i1 = water_window(z)
    assert i1 >= z.size - 8                           # hover retained


def test_extend_z_by_w_fills_truncated_ctd_tail():
    import numpy as np

    from ladcp.qa.depth import _extend_z_by_w
    # CTD record stops mid-ascent (CRUISE2 t1-01); ADCP w keeps going
    tad = np.arange(300, dtype=float)
    z = np.full(300, np.nan)
    z[:200] = np.concatenate([np.linspace(5, 100, 100), np.linspace(100, 52, 100)])
    w_true = np.gradient(z[:200], tad[:200])          # dz/dt over the known part
    w_ad = np.full(300, np.nan)
    w_ad[:200] = w_true
    w_ad[200:] = -0.48                                # keeps ascending at ~0.5 m/s
    out = _extend_z_by_w(z, w_ad, tad)
    assert np.isfinite(out[200:300]).sum() > 80       # tail filled...
    assert out[250] == pytest.approx(52 - 0.48 * 51, abs=2.0)
    fin = np.where(np.isfinite(out))[0]
    assert (out[fin] >= 2.0 - 1e-9).all()             # ...and stops at the surface


def test_extend_z_by_w_skips_when_sign_uncertain():
    import numpy as np

    from ladcp.qa.depth import _extend_z_by_w
    tad = np.arange(100, dtype=float)
    z = np.full(100, np.nan)
    z[:50] = np.linspace(5, 50, 50)
    rng = np.random.default_rng(0)
    w_ad = rng.normal(0, 0.1, 100)                    # uncorrelated noise -> no sign
    out = _extend_z_by_w(z, w_ad, tad)
    assert not np.isfinite(out[50:]).any()
