"""Sound-speed correction (legacy p.soundcorr) + shear-vs-inverse consistency WARN.

Both ported 2026-06-13 from the deep legacy-vs-pyladcp audit:

* **soundcorr** (prepinv.m:35,419-444, ON by default): RDI Doppler velocity is
  proportional to the fixed firmware sound speed; when that value is wrong every
  cell is biased by ``c_in-situ/c_firmware``. We rescale ru/rv/rw per ensemble
  before super-ensemble formation. On MORIA-80 the firmware ran a fixed 1450 m/s
  vs an in-situ median ~1492 -> a ~2.9 % scale, visible as the golden ubar deficit
  (-6.31 -> -6.47 vs golden -6.50; bias +0.20 -> +0.04 cm/s).
* **shear-vs-inverse consistency** (getshear2.m:143-158): the inverse and the shear
  method estimate the same baroclinic shape; when they disagree by more than the
  formal error, legacy inflates the delivered uerr (mean -> uvds/1.5) and WARNs.
"""

from __future__ import annotations

import pathlib
from dataclasses import replace

import numpy as np
import pytest

from ladcp.config import moria05_params
from ladcp.io.ctd_cnv import read_ctd_cnv
from ladcp.qa.checks import consistency_checks
from ladcp.qa.ingest import load_dualhead
from ladcp.qa.inverse import (
    ShearProfile,
    VelocityProfile,
    _insitu_soundspeed,
    _shear_inverse_consistency,
    _soundspeed_scale,
    build_solve_context,
    compute_velocity_full,
)

ROOT = pathlib.Path(__file__).resolve().parent / "fixtures"
GOOD = ROOT / "New_golden" / "Good"
DOWN = GOOD / "LADCP" / "MORIA-80-LADCP-M.000"
UP = GOOD / "LADCP" / "MORIA-80-LADCP-S.000"
CTD = GOOD / "CTD" / "moria-80_clean.cnv"
DROT = -9.878379

needs_fixtures = pytest.mark.skipif(not DOWN.exists(), reason="MORIA New_golden not present")


def _load(params=None):
    p = params if params is not None else moria05_params()
    dh = load_dualhead(str(DOWN), str(UP), station="MORIA-80", params=p)
    ctd = read_ctd_cnv(str(CTD), params=p)
    return dh, ctd, p


# --------------------------------------------------------------------------- soundcorr

@needs_fixtures
def test_soundspeed_scale_is_firmware_vs_insitu_ratio():
    dh, ctd, p = _load()
    from ladcp.qa.depth import synchronize
    sync = synchronize(dh, ctd)
    n = min(dh.down.n_ens, dh.up.n_ens)
    sc = _soundspeed_scale(dh, ctd, sync, n)
    assert sc is not None and sc.shape == (n,)
    # firmware fixed 1450 vs in-situ median ~1492 -> ~+2.9 %, well inside the +-15 % clamp
    med = float(np.nanmedian(sc))
    assert 1.02 < med < 1.05, med
    assert np.all(sc >= 0.85) and np.all(sc <= 1.15)


@needs_fixtures
def test_soundcorr_moves_ubar_toward_golden():
    """ON (default) scales |ubar| up ~2.9 % vs OFF, toward golden -6.50 cm/s."""
    dh, ctd, p_on = _load()
    p_off = replace(p_on, soundcorr=False)
    r_on = compute_velocity_full(dh, ctd, drot=DROT, params=p_on, solver="inverse")
    r_off = compute_velocity_full(dh, ctd, drot=DROT, params=p_off, solver="inverse")
    # correction amplifies the barotropic magnitude by ~the sound-speed ratio
    ratio = r_on.vp.ubar / r_off.vp.ubar
    assert 1.02 < ratio < 1.05, (r_on.vp.ubar, r_off.vp.ubar, ratio)
    # and lands close to golden -0.0650 m/s (was ~-0.0631 uncorrected)
    assert abs(r_on.vp.ubar - (-0.0650)) < abs(r_off.vp.ubar - (-0.0650))


@needs_fixtures
def test_soundcorr_flag_disables():
    dh, ctd, _ = _load()
    p_off = replace(moria05_params(), soundcorr=False)
    se_off = build_solve_context(dh, ctd, dz=8.0, params=p_off)[2]
    se_on = build_solve_context(dh, ctd, dz=8.0, params=moria05_params())[2]
    # with the correction off the merged water velocities are left unscaled
    fin = np.isfinite(se_off.ru) & np.isfinite(se_on.ru)
    assert fin.any()
    assert not np.allclose(se_off.ru[fin], se_on.ru[fin])


@needs_fixtures
def test_insitu_soundspeed_none_without_salinity():
    _, ctd, _ = _load()
    ctd_nosal = replace(ctd, salinity=None)
    assert _insitu_soundspeed(ctd_nosal, np.array([100.0, 200.0])) is None


# --------------------------------------------------------------- shear vs inverse

def _vp(u, v, uerr):
    z = np.arange(8.0, 8.0 * (len(u) + 1), 8.0)
    n = np.full(len(u), 5)
    return VelocityProfile(z=z, u=np.asarray(u, float), v=np.asarray(v, float),
                           uerr=np.asarray(uerr, float), nvel=n, ubar=0.0, vbar=0.0, n=n)


def _shear(u, v):
    z = np.arange(8.0, 8.0 * (len(u) + 1), 8.0)
    zero = np.zeros(len(u))
    return ShearProfile(z=z, u=np.asarray(u, float), v=np.asarray(v, float), w=zero,
                        u_shear=zero, v_shear=zero, n=np.full(len(u), 5))


def test_shear_inverse_consistency_zero_when_identical():
    u = np.array([0.1, 0.2, 0.3, 0.2, 0.1])
    v = np.array([0.0, -0.1, -0.2, -0.1, 0.0])
    uvds, mean_uerr = _shear_inverse_consistency(_vp(u, v, np.full(5, 0.03)), _shear(u, v))
    assert uvds < 1e-9
    assert mean_uerr == pytest.approx(0.03)


def test_shear_inverse_consistency_nonzero_when_different():
    u = np.array([0.1, 0.2, 0.3, 0.2, 0.1])
    v = np.array([0.0, -0.1, -0.2, -0.1, 0.0])
    u2 = u + np.array([0.0, 0.2, -0.2, 0.2, -0.2])      # different baroclinic shape
    uvds, _ = _shear_inverse_consistency(_vp(u, v, np.full(5, 0.03)), _shear(u2, v))
    assert uvds > 0.1


def test_shear_inverse_grid_mismatch_returns_none():
    assert _shear_inverse_consistency(_vp([0.1, 0.2], [0, 0], [0.03, 0.03]),
                                      _shear([0.1, 0.2, 0.3], [0, 0, 0])) is None


@needs_fixtures
def test_shear_inverse_metric_present_and_ok_on_clean_cast():
    dh, ctd, p = _load()
    r = compute_velocity_full(dh, ctd, drot=DROT, params=p, solver="inverse")
    assert r.shear_inverse is not None
    names = {m.name: m for m in consistency_checks(r)}
    assert "shear_vs_inverse_consistency" in names
    # MORIA-80 is a clean dual-head cast: shear and inverse agree (no WARN)
    from ladcp.models import Status
    assert names["shear_vs_inverse_consistency"].status == Status.OK


@needs_fixtures
def test_shear_inverse_none_for_shear_solver():
    dh, ctd, p = _load()
    r = compute_velocity_full(dh, ctd, drot=DROT, params=p, solver="shear")
    assert r.shear_inverse is None
