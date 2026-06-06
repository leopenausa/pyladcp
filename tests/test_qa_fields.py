"""Figure 3 (super-ensemble error) + drift-map data and figures."""

from __future__ import annotations

import pathlib

import numpy as np
import pytest

from ladcp.qa.inverse import DriftTrack, ErrField, VelocityProfile, VelocityResult

ROOT = pathlib.Path(__file__).resolve().parent / "fixtures"
GOOD = ROOT / "New_golden" / "Good"
DOWN = GOOD / "LADCP" / "MORIA-80-LADCP-M.000"
UP = GOOD / "LADCP" / "MORIA-80-LADCP-S.000"
CTD = GOOD / "CTD" / "moria-80_clean.cnv"


def _vp():
    z = np.arange(20.0, 200.0, 10.0)
    return VelocityProfile(z=z, u=np.linspace(0.2, -0.1, z.size),
                           v=np.full_like(z, -0.05), uerr=np.full_like(z, 0.02),
                           nvel=np.full_like(z, 5), ubar=0.05, vbar=-0.05,
                           n=np.full_like(z, 5))


def _err_field(nbin=20, nse=30):
    rng = np.random.default_rng(0)
    depth = np.linspace(20, 400, nbin)[:, None] + np.zeros((nbin, nse))
    se_index = np.broadcast_to(np.arange(1, nse + 1), (nbin, nse)).astype(float)
    resid = rng.normal(0, 0.02, (nbin, nse))
    return ErrField(se_index=se_index, depth=depth, resid_u=resid, resid_v=resid.copy(),
                    u_oce=np.full((nbin, nse), 0.1), v_oce=np.full((nbin, nse), -0.05),
                    binno=np.arange(-nbin // 2, nbin // 2), u_std=0.02, v_std=0.02)


def _drift():
    t = np.linspace(0, 1, 50)
    pkg_e = -100 * np.sin(np.pi * t)            # loops out and back
    return DriftTrack(ship_e=np.zeros(50), ship_n=np.zeros(50),
                      ship_sog=np.full(50, 0.1), pkg_e=pkg_e, pkg_n=pkg_e * 0.3,
                      i_bottom=25)


def _result(err=None, drift=None):
    return VelocityResult(vp=_vp(), bp=None, shear=None, zbottom=300.0,
                          resid_z=np.array([]), resid_u=np.array([]), resid_v=np.array([]),
                          err=err, drift=drift)


# --------------------------------------------------------------------------- #
# figures render from synthetic data
# --------------------------------------------------------------------------- #
def test_error_figure_renders():
    import matplotlib
    matplotlib.use("Agg")
    from ladcp.plots.error_figure import error_figure

    assert error_figure(_result(err=_err_field()), station="T") is not None
    assert error_figure(_result(err=None)) is not None        # placeholder, no error


def test_drift_figure_renders():
    import matplotlib
    matplotlib.use("Agg")
    from ladcp.plots.drift_figure import drift_figure

    assert drift_figure(_result(drift=_drift()), station="T") is not None
    assert drift_figure(_result(drift=None)) is not None


# --------------------------------------------------------------------------- #
# real-data integration (skipped without the fixture)
# --------------------------------------------------------------------------- #
@pytest.mark.skipif(not DOWN.exists(), reason="MORIA-80 fixture not present")
def test_err_and_drift_populated_and_closed():
    from ladcp.config import moria05_params
    from ladcp.io.ctd_cnv import read_ctd_cnv
    from ladcp.qa.ingest import load_dualhead
    from ladcp.qa.inverse import compute_velocity_full

    p = moria05_params()
    dh = load_dualhead(str(DOWN), str(UP), station="MORIA-80", params=p)
    ctd = read_ctd_cnv(str(CTD), params=p)
    r = compute_velocity_full(dh, ctd, drot=-5.44, params=p, solver="inverse")

    # error field: robust residual std is small (well-fit inverse)
    assert r.err is not None
    assert r.err.depth.shape[1] > 50           # many super-ensembles
    assert 0.0 < r.err.u_std < 0.06 and 0.0 < r.err.v_std < 0.06

    # drift: package track closes on the ship's end position (loop closure)
    d = r.drift
    assert d is not None and d.pkg_e.size > 50
    assert abs(d.pkg_e[-1] - d.ship_e[-1]) < 1e-6
    assert abs(d.pkg_n[-1] - d.ship_n[-1]) < 1e-6
    assert 0 < d.i_bottom < d.pkg_e.size - 1    # turnaround is mid-cast
