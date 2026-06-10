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
    z = np.arange(1, 7) * 8.0                   # 8..48 m: a genuine near-surface gap
    _surface_fill(u, v, uerr, nvel, z)
    assert u[0] == u[1] == u[2] == -0.20        # bins above the first reliable bin filled
    assert v[0] == v[1] == -0.20
    assert np.allclose(u[2:], [-0.20, -0.21, -0.22, -0.23])   # deeper bins untouched


def test_surface_fill_leaves_large_gap_nan():
    # first reliable bin is ~1000 m down (ADCP began mid-descent): do NOT fabricate a flat top
    from ladcp.qa.inverse_full import _surface_fill
    nz = 600
    u = np.full(nz, np.nan); u[125:] = -0.20    # data only below ~1000 m
    v = u.copy(); uerr = np.full(nz, 0.03)
    nvel = np.zeros(nz, dtype=int); nvel[125:] = 40
    z = np.arange(1, nz + 1) * 8.0              # bin 125 ~ 1008 m  >> _SURFACE_FILL_MAX
    _surface_fill(u, v, uerr, nvel, z)
    assert np.all(np.isnan(u[:125]))            # upper water column left unsampled, not filled
    assert np.allclose(u[125:], -0.20)


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
    # The solver switch itself does not perturb the shear shape (corr stays ~0.998). The absolute
    # rms reflects the velocity-path editing fix set now applied before averaging
    # (units/floor/refmed/btgate/outlier_n): MORIA-80 shear rms is bias-dominated while the
    # profile shape is unchanged. See memory ladcp-editing-rootcause-2026-06.
    # 2026-06-10 (FDCCC1 reference-offset root-cause): the bottom track now prefers the RDI
    # firmware track when the PD0 carries one -- legacy btrk_mode=3 semantics (legacy used it
    # on MORIA too, btrk_used=1). MORIA-80's RDI and own tracks differ ~1 cm/s, moving the
    # BT-referenced shear bias 0.031 -> 0.0397; the inverse (the product path) is golden-
    # validated unchanged by the tighter tests above.
    out, _ = both_solvers
    _, s = out["shear"]
    assert s["u"].corr > 0.997
    assert s["u"].rms < 0.045


def test_lainsadcp_weight_math():
    # row weight == sadcpfac*velerr/verr; rhs == d*w; jz == clip(round(z/dz))
    from ladcp.qa.inverse_full import _lainsadcp
    svel = np.array([[40., 0.10, -0.05, 0.05],
                     [80., 0.12, -0.06, 0.10],
                     [120., 0.08, -0.04, 0.08],
                     [160., 0.05, -0.02, 0.04]])
    jz, w, rhs, kept = _lainsadcp(svel, nz=30, dz=8.0, sadcpfac=3.0, velerr=0.06)
    assert np.allclose(w, 3.0 * 0.06 / svel[:, 3])
    assert np.allclose(rhs, (svel[:, 1] + 1j * svel[:, 2]) * w)
    assert np.array_equal(jz, np.clip(np.round(svel[:, 0] / 8.0).astype(int), 1, 30) - 1)


def test_lainsadcp_reproduces_fdccc1_002_golden_weight():
    # the embedded SADCP profiles + golden velerr must give the logged mean weight 1.5993
    from ladcp.qa import validate as V
    from ladcp.qa.inverse_full import _lainsadcp
    if not V.STATIONS["FDCCC1_002"].mat.exists():
        pytest.skip("FDCCC1_002 golden not present (local-only)")
    dr = V.load_dr("FDCCC1_002")
    svel = np.column_stack([dr.z_sadcp, dr.u_sadcp, dr.v_sadcp, dr.uerr_sadcp])
    svel = svel[np.isfinite(svel[:, 1])]
    _, w, _, _ = _lainsadcp(svel, nz=int(225 / 8), dz=8.0, sadcpfac=3.0, velerr=0.056327)
    assert np.mean(w) == pytest.approx(1.5993, abs=1e-3)


def test_sadcp_pulls_solution(both_solvers):
    # injecting a +0.15 m/s SADCP offset shifts the inverse toward it, scaling with sadcpfac
    p = moria05_params()
    dh = load_dualhead(str(DOWN), str(UP), station="MORIA-80", params=p)
    ctd = read_ctd_cnv(str(CTD), params=p)
    base = compute_velocity_full(dh, ctd, drot=DROT, params=p, solver="inverse").vp
    zs = np.array([40., 80., 120., 160., 200.])
    svel = np.column_stack([zs, np.interp(zs, base.z, base.u) + 0.15,
                            np.interp(zs, base.z, base.v), np.full(zs.size, 0.05)])
    band = (base.z >= 32) & (base.z <= 208)

    def du(fac):
        vp = compute_velocity_full(dh, ctd, drot=DROT, params=p, solver="inverse",
                                   sadcp=svel, sadcpfac=fac).vp
        return float(np.nanmean(vp.u[band] - base.u[band]))

    d1, d3, d10 = du(1.0), du(3.0), du(10.0)
    assert 0 < d1 < d3 < d10          # monotonic pull toward the +0.15 target
    assert d10 > 0.04


def test_bad_solver_name_rejected():
    p = moria05_params()
    dh = load_dualhead(str(DOWN), str(UP), station="MORIA-80", params=p)
    ctd = read_ctd_cnv(str(CTD), params=p)
    with pytest.raises(ValueError, match="solver"):
        compute_velocity_full(dh, ctd, drot=DROT, params=p, solver="bogus")
