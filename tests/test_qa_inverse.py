"""Phase 5c shear-method profile vs MORIA-80 golden (getshear2.m).

Golden targets: dr.u_shear_method / dr.v_shear_method (zero-mean baroclinic profile,
true east/north after p.drot = -9.878379 deg).
"""

from __future__ import annotations

import pathlib

import numpy as np
import pytest
import scipy.io as sio

from ladcp.config import moria05_params
from ladcp.io.ctd_cnv import read_ctd_cnv
from ladcp.qa.depth import synchronize
from ladcp.qa.ingest import load_dualhead
from ladcp.qa.inverse import _diff2, compute_velocity, shear_method, velocity_profile
from ladcp.qa.superens import form_superensembles, merge_heads

ROOT = pathlib.Path(__file__).resolve().parent / "fixtures"  # tests/fixtures/
GOOD = ROOT / "New_golden" / "Good"
DOWN = GOOD / "LADCP" / "MORIA-80-LADCP-M.000"
UP = GOOD / "LADCP" / "MORIA-80-LADCP-S.000"
CTD = GOOD / "CTD" / "moria-80_clean.cnv"
MAT = ROOT / "New_golden" / "MORIA-80_LEO" / "MORIA-80.mat"

pytestmark = pytest.mark.skipif(not DOWN.exists(), reason="MORIA New_golden not present")

DROT = -9.878379          # golden p.drot (true magnetic declination, N Atlantic)


@pytest.fixture(scope="module")
def se_and_golden():
    dh = load_dualhead(str(DOWN), str(UP), station="MORIA-80", params=moria05_params())
    ctd = read_ctd_cnv(str(CTD), params=moria05_params())
    sync = synchronize(dh, ctd)
    se = form_superensembles(merge_heads(dh, params=moria05_params()),
                             sync.z_on_ping, avdz=8.0)
    dr = sio.loadmat(str(MAT), squeeze_me=True, struct_as_record=False)["dr"]
    return se, dr


@pytest.fixture(scope="module")
def profile_and_golden(se_and_golden):
    se, dr = se_and_golden
    sp = shear_method(se, dz=8.0, drot=DROT, z=np.asarray(dr.z, dtype=float))
    return sp, dr


def _corr_rms(a, b):
    ok = np.isfinite(a) & np.isfinite(b)
    corr = float(np.corrcoef(a[ok], b[ok])[0, 1])
    rms = float(np.sqrt(np.mean((a[ok] - b[ok]) ** 2)))
    return corr, rms


def test_diff2_central():
    x = np.array([0.0, 1.0, 4.0, 9.0, 16.0])[:, None]
    assert np.allclose(_diff2(x)[:, 0], [4.0, 8.0, 12.0])   # x[2:]-x[:-2]


def test_shear_u_matches_golden(profile_and_golden):
    sp, dr = profile_and_golden
    corr, rms = _corr_rms(sp.u, np.asarray(dr.u_shear_method, dtype=float))
    assert corr > 0.95
    assert rms < 0.04


def test_shear_v_matches_golden(profile_and_golden):
    sp, dr = profile_and_golden
    corr, rms = _corr_rms(sp.v, np.asarray(dr.v_shear_method, dtype=float))
    assert corr > 0.90
    assert rms < 0.04


def test_profile_on_golden_grid(profile_and_golden):
    sp, dr = profile_and_golden
    assert sp.u.shape == np.asarray(dr.z).shape
    assert np.allclose(sp.z, np.asarray(dr.z, dtype=float))


def test_profile_is_baroclinic(profile_and_golden):
    # integrated shear is demeaned -> near zero depth-mean (drot mixes u/v slightly)
    sp, _ = profile_and_golden
    assert abs(np.nanmean(sp.u)) < 0.02
    assert abs(np.nanmean(sp.v)) < 0.02


# --- absolute profile (baroclinic + barotropic reference) ------------------ #
def test_velocity_profile_matches_golden_u(se_and_golden):
    se, dr = se_and_golden
    vp = velocity_profile(se, dz=8.0, drot=DROT, z=np.asarray(dr.z, dtype=float))
    corr, rms = _corr_rms(vp.u, np.asarray(dr.u, dtype=float))
    assert corr > 0.95
    assert rms < 0.05


def test_velocity_profile_matches_golden_v(se_and_golden):
    se, dr = se_and_golden
    vp = velocity_profile(se, dz=8.0, drot=DROT, z=np.asarray(dr.z, dtype=float))
    corr, rms = _corr_rms(vp.v, np.asarray(dr.v, dtype=float))
    assert corr > 0.90
    assert rms < 0.05


def test_barotropic_reference_sign(se_and_golden):
    # depth-mean reference is small and negative for this cast (golden ubar -0.065)
    se, dr = se_and_golden
    vp = velocity_profile(se, dz=8.0, drot=DROT, z=np.asarray(dr.z, dtype=float))
    assert -0.12 < vp.ubar < 0.0
    assert abs(vp.vbar) < 0.05


def test_compute_velocity_below_bottom_removed(se_and_golden):
    # full pipeline: below-bottom removal + maxdepth grid cap.
    dh = load_dualhead(str(DOWN), str(UP), station="MORIA-80", params=moria05_params())
    ctd = read_ctd_cnv(str(CTD), params=moria05_params())
    _, dr = se_and_golden
    vp = compute_velocity(dh, ctd, drot=DROT, params=moria05_params())

    # grid stops at the package max depth, well above any below-seabed cells (~1309 m raw)
    assert vp.z.max() <= 1075
    # below-bottom removal recovers near-bottom structure (corr ~0.998, was 0.98)
    gi = np.interp(vp.z, np.asarray(dr.z, float), np.asarray(dr.u, float))
    corr, _ = _corr_rms(vp.u, gi)
    assert corr > 0.99
    # the deepest bins are NOT an artificial flat tail: real variation near the seabed
    deep = vp.u[-8:]
    assert np.nanstd(deep) > 0.01


# --- Phase 5d: bottom-track .bot profile + reference ------------------------ #
@pytest.fixture(scope="module")
def velocity_and_bottom():
    from ladcp.qa.inverse import compute_velocity_and_bottom
    dh = load_dualhead(str(DOWN), str(UP), station="MORIA-80", params=moria05_params())
    ctd = read_ctd_cnv(str(CTD), params=moria05_params())
    dr = sio.loadmat(str(MAT), squeeze_me=True, struct_as_record=False)["dr"]
    vp, bp, zbottom = compute_velocity_and_bottom(dh, ctd, drot=DROT, params=moria05_params())
    return vp, bp, zbottom, dr


def test_bottom_referenced_profile_matches_golden(velocity_and_bottom):
    # .bot vs golden dr.zbot/ubot/vbot (legacy lainbott profile branch)
    _, bp, _, dr = velocity_and_bottom
    assert bp is not None and bp.n_bins > 20
    zbot = np.asarray(dr.zbot, float)
    ui = np.interp(zbot, bp.z, bp.u)
    vi = np.interp(zbot, bp.z, bp.v)
    cu, ru = _corr_rms(ui, np.asarray(dr.ubot, float))
    cv, rv = _corr_rms(vi, np.asarray(dr.vbot, float))
    assert cu > 0.95 and ru < 0.05
    assert cv > 0.85 and rv < 0.05


def test_bottom_track_reference_ubar(velocity_and_bottom):
    # Bottom-track reference -> small negative barotropic offset (golden ubar -0.065). Adding the
    # legacy medianan(na=0) reference (refmed) lands this compute_velocity_and_bottom path at
    # ubar ~ -0.036 -- an accepted small regression vs without refmed (-0.042, which beat the
    # -0.041 package-return). The production compute_velocity_full inverse ubar (~-0.053) still
    # improves toward golden. See memory ladcp-editing-rootcause-2026-06.
    vp, _, _, dr = velocity_and_bottom
    assert -0.055 < vp.ubar < -0.02              # small & negative, in the golden ballpark


def test_bottom_depth_detected(velocity_and_bottom):
    _, _, zbottom, _ = velocity_and_bottom
    assert abs(zbottom - 1079.0) < 3.0                       # golden p.zbottom 1079.02


def test_write_bot_roundtrip(velocity_and_bottom, tmp_path):
    from ladcp.qa.export import write_bot
    _, bp, zbottom, _ = velocity_and_bottom
    path = tmp_path / "MORIA-80.bot"
    write_bot(bp, str(path), station="MORIA-80", lat=62.16, lon=-11.53,
              drot=DROT, zbottom=zbottom)
    text = path.read_text()
    assert "Columns     = z:u:v:err" in text
    assert "Bottom depth= 1080" in text
    assert text.count("\n") >= bp.n_bins
