"""End-to-end regression for the large clock-offset sync fix + near-touch false-lock guard.

CRUISE2_001 is the canonical failure case: the ADCP pinged ~24 min on deck before the cast, so
the in-water window sits ~2400 s into the record -- beyond the old single-stage ``bestlag``
reach. The old pipeline mis-mapped the depth onto the on-deck pings (0% velocity coverage) and
the profile collapsed to empty; the seabed stack then locked a deep constant-range artifact
(242 m, golden 127 m). This guards both fixes on the real cast. Local-only data -> skipped on CI.
"""

from __future__ import annotations

import numpy as np
import pytest

from ladcp.config import resolve_params
from ladcp.io.ctd_cnv import read_ctd_cnv
from ladcp.qa import validate as V
from ladcp.qa.bottom import detect_bottom
from ladcp.qa.depth import synchronize
from ladcp.qa.ingest import apply_header_config, load_dualhead
from ladcp.qa.inverse import compute_velocity_full

_ST = V.STATIONS["CRUISE2_001"]
pytestmark = pytest.mark.skipif(not _ST.has_raw, reason="CRUISE2_001 raw not present (local-only)")


@pytest.fixture(scope="module")
def fdccc1_001():
    p = resolve_params("MORIA", "CRUISE2_001")     # geometry comes from the PD0 headers
    dh = load_dualhead(str(_ST.raw_down), str(_ST.raw_up), station="CRUISE2_001", params=p)
    apply_header_config(p, dh)
    ctd = read_ctd_cnv(str(_ST.ctd), params=p)
    return dh, ctd, p


def test_sync_locates_buried_cast(fdccc1_001):
    dh, ctd, _ = fdccc1_001
    sync = synchronize(dh, ctd)
    assert sync.lag > 1500                          # the cast is deep in the recording
    assert sync.coarse_score > 0.5                  # confidently located
    assert sync.maxdepth == pytest.approx(125.0, abs=3.0)
    # the in-water window must now overlap the valid-velocity pings (was ~0 before the fix)
    n = dh.down.n_ens
    from ladcp.qa.depth import water_window
    i0, i1 = water_window(sync.z_on_ping[:n])
    good = (np.isfinite(dh.down.vel[0]).sum(axis=0) >= 3)[:n]
    coverage = good[i0:i1 + 1].mean()
    assert coverage > 0.5


def test_near_touch_seabed_not_deep_artifact(fdccc1_001):
    dh, ctd, _ = fdccc1_001
    b = detect_bottom(dh, synchronize(dh, ctd), ctd=ctd)
    # golden seabed 127 m; the package nearly touched (zmax ~125), so the honest answer is the
    # flagged near-touch lower bound -- NOT the 242 m constant-range artifact the bare stack locks.
    assert b.is_fallback
    assert b.zbottom == pytest.approx(126.0, abs=6.0)


def test_profile_is_non_empty_and_matches_golden(fdccc1_001):
    dh, ctd, p = fdccc1_001
    vr = compute_velocity_full(dh, ctd, drot=0.0, params=p, solver="inverse")
    dr = V.load_dr("CRUISE2_001")
    sc = V.score_profile(vr.vp.z, vr.vp.u, vr.vp.v, dr)
    assert sc["u"].n >= 10                           # was 0 (empty) before the fix
    assert sc["u"].corr > 0.9                        # legacy medianan(na=0) reference lifts this
    #                                                  shallow near-touch full inverse to ~0.97
