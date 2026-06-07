"""MORIA-10 end-to-end validation (#4a/#2): raw VmDAS archive -> velocity vs golden.

MORIA-10 (Gulf of Biscay, 549 m, Nav+Bottom) is the first cross-validation station unlocked
by the #4a data-driven ingest: its raw inputs are the VmDAS archive (master MLADC012 + slave
SLADC013, paired by time) rather than curated per-station files. We score the computed
profile against the golden ``dr`` using the golden's drot (its legacy IGRF00 frame), so the
test measures solver quality, not the magdec offset. Skips when the raw archive is absent.
"""

from __future__ import annotations

import numpy as np
import pytest

from ladcp.config import resolve_params
from ladcp.io.ctd_cnv import read_ctd_cnv
from ladcp.qa import validate as V
from ladcp.qa.ingest import apply_header_config, load_dualhead
from ladcp.qa.inverse import compute_velocity_full

ST = V.STATIONS["MORIA-10"]
pytestmark = pytest.mark.skipif(not ST.has_raw, reason="MORIA-10 raw archive not present")


@pytest.fixture(scope="module")
def results():
    p = resolve_params("MORIA", "MORIA-10")
    dh = load_dualhead(str(ST.raw_down), str(ST.raw_up), station="MORIA-10", params=p)
    apply_header_config(p, dh)
    ctd = read_ctd_cnv(str(ST.ctd), params=p)
    dr = V.load_dr("MORIA-10")
    out = {}
    for solver in ("shear", "inverse"):
        r = compute_velocity_full(dh, ctd, drot=ST.drot, params=p, solver=solver)
        out[solver] = (r, V.score_profile(r.vp.z, r.vp.u, r.vp.v, dr))
    return out, dr


def test_seabed_matches_golden(results):
    out, _ = results
    r, _ = out["inverse"]
    assert abs(r.zbottom - 558.25) < 5.0            # golden p.zbottom 558.2497


@pytest.mark.parametrize("solver", ["shear", "inverse"])
def test_profile_tracks_golden(results, solver):
    out, _ = results
    _, s = out[solver]
    assert s["u"].corr > 0.88 and s["u"].rms < 0.045
    assert s["v"].corr > 0.85 and s["v"].rms < 0.050
    assert s["u"].n > 50                            # scored over the full golden grid


@pytest.mark.parametrize("solver", ["shear", "inverse"])
def test_barotropic_reference_near_golden(results, solver):
    out, dr = results
    r, _ = out[solver]
    assert abs(r.vp.ubar - dr.ubar) < 0.03          # golden ubar -0.007
    assert abs(r.vp.vbar - dr.vbar) < 0.03          # golden vbar +0.111


def test_slave_paired_by_time_not_index():
    # the discovery layer must pick SLADC013 (time-overlap), not the same-index SLADC012
    assert ST.raw_up.name == "SLADC013.000"
    assert ST.raw_down.name == "MLADC012.000"
    assert np.isfinite(ST.drot)
