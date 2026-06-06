"""Full constrained inverse (solver="inverse") vs MORIA-80 golden.

The inverse forms the getinv.m weighted L2 solve (data + smoothing + bottom-track +
navigation constraints) on the live qa.SuperEns. It must reproduce the golden dr.u/dr.v
at profile level -- on par with the validated shear path -- and pin the barotropic
reference near the golden ubar (-0.065, set by the bottom track on this static-DP cast).
"""

from __future__ import annotations

import pathlib

import numpy as np
import pytest

from ladcp.config import moria05_params
from ladcp.io.ctd_cnv import read_ctd_cnv
from ladcp.qa import validate as V
from ladcp.qa.ingest import load_dualhead
from ladcp.qa.inverse import compute_velocity_full

ROOT = pathlib.Path(__file__).resolve().parent / "fixtures"
GOOD = ROOT / "New_golden" / "Good"
DOWN = GOOD / "LADCP" / "MORIA-80-LADCP-M.000"
UP = GOOD / "LADCP" / "MORIA-80-LADCP-S.000"
CTD = GOOD / "CTD" / "moria-80_clean.cnv"

pytestmark = pytest.mark.skipif(not DOWN.exists(), reason="MORIA New_golden not present")
DROT = -9.878379


@pytest.fixture(scope="module")
def both_solvers():
    p = moria05_params()
    dh = load_dualhead(str(DOWN), str(UP), station="MORIA-80", params=p)
    ctd = read_ctd_cnv(str(CTD), params=p)
    dr = V.load_dr("MORIA-80")
    out = {}
    for solver in ("shear", "inverse"):
        r = compute_velocity_full(dh, ctd, drot=DROT, params=p, solver=solver)
        out[solver] = (r.vp, V.score_profile(r.vp.z, r.vp.u, r.vp.v, dr))
    return out, dr


def test_surface_fill_repairs_collapsed_top():
    # the two shallowest bins are sparse/collapsed (~0); fill from the first reliable bin
    from ladcp.qa.inverse_full import _surface_fill
    u = np.array([0.0, 0.0, -0.20, -0.21, -0.22, -0.23])
    v = u.copy()
    uerr = np.full(6, 0.03)
    nvel = np.array([2, 5, 40, 42, 45, 50])     # median(>0)=43 -> thr=max(3,0.4*43)=17
    _surface_fill(u, v, uerr, nvel)
    assert u[0] == u[1] == u[2] == -0.20        # bins above the first reliable bin filled
    assert v[0] == v[1] == -0.20
    assert np.allclose(u[2:], [-0.20, -0.21, -0.22, -0.23])   # deeper bins untouched


def test_inverse_matches_golden_u(both_solvers):
    out, _ = both_solvers
    _, s = out["inverse"]
    assert s["u"].corr > 0.97
    assert s["u"].rms < 0.03


def test_inverse_matches_golden_v(both_solvers):
    out, _ = both_solvers
    _, s = out["inverse"]
    assert s["v"].corr > 0.93
    assert s["v"].rms < 0.03


def test_inverse_reference_near_golden(both_solvers):
    # bottom-track-pinned barotropic reference; golden ubar -0.065
    out, dr = both_solvers
    vp, _ = out["inverse"]
    assert abs(vp.ubar - dr.ubar) < 0.03
    assert abs(vp.vbar) < 0.05


def test_inverse_grid_stops_at_seabed(both_solvers):
    # the inverse must not grid below the seabed (zbottom ~1080 m)
    out, _ = both_solvers
    vp, _ = out["inverse"]
    assert vp.z[-1] < 1080.0
    assert np.all(np.diff(vp.z) > 0)


def test_shear_path_unchanged(both_solvers):
    # adding the solver switch must not perturb the validated shear solution
    out, _ = both_solvers
    _, s = out["shear"]
    assert s["u"].corr > 0.997
    assert s["u"].rms < 0.02


def test_bad_solver_name_rejected():
    p = moria05_params()
    dh = load_dualhead(str(DOWN), str(UP), station="MORIA-80", params=p)
    ctd = read_ctd_cnv(str(CTD), params=p)
    with pytest.raises(ValueError, match="solver"):
        compute_velocity_full(dh, ctd, drot=DROT, params=p, solver="bogus")
