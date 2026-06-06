"""Phase-4 constrained inverse (getinv.m) — ocean velocity profile.

Validates the Nav+bottom-track inverse against the golden gridded ``u``/``v`` on the quiet
stations (016, 117). The strong, well-constrained result is the **absolute reference**: the
depth-averaged (barotropic) velocity matches the golden to <1.5 cm/s, and the profile-mean
signed difference is sub-cm/s — i.e. the data / smoothing / barotropic / bottom-track
machinery is correct. The point-wise shape scatter (~5-7 cm/s) is checked loosely: it
reflects the omitted SADCP constraint (the golden's reference combo is Nav+BT+SADCP; SADCP is
a later phase) and the upstream loadrdi near-field fidelity gap (see
[[ladcp-editdata-pglim-findings]]), and is at the level the LDEO lineage shows without full
fidelity. It is independent of ``velerr`` (0.03/0.07/0.09 all give the same shape).
"""

from __future__ import annotations

import json
import pathlib

import numpy as np
import pytest
import xarray as xr

from ladcp.ix.bottom import own_bottom_track
from ladcp.ix.depth import ctd_depth_on_ping, detect_bottom_and_flag
from ladcp.ix.inverse import _lainsmoo, _lanarrow_reject, _solve, invert
from ladcp.ix.superens import form_superensembles
from ladcp.io.ctd_seabird import load_seabird_cnv

DATA = pathlib.Path(__file__).resolve().parents[1] / "test_data" / "goship" / "RB1606_P18"
EXPECTED = pathlib.Path(__file__).resolve().parents[1] / "test_data" / "goship" / "expected_values.json"
CNV = {"016": "ctd_009_01.cnv", "117": "ctd_109_01.cnv"}
STATIONS = ["016", "117"]

pytestmark = pytest.mark.skipif(
    not (DATA / "016" / "016DL000.000").exists(),
    reason="GO-SHIP raw PD0 not present",
)


def _expected():
    return json.loads(EXPECTED.read_text())


def _run(st: str):
    from ladcp.ix.ingest import load_ladcp
    ds = xr.open_dataset(str(DATA / st / f"{st}.nc"))
    a = ds.attrs
    ts = np.asarray(a["time_start"], int)
    te = np.asarray(a["time_end"], int)
    f = "%04d-%02d-%02dT%02d:%02d:%02d"
    t0, t1 = np.datetime64(f % tuple(ts)), np.datetime64(f % tuple(te))
    d = load_ladcp(str(DATA / st / f"{st}DL000.000"), str(DATA / st / f"{st}UL000.000"),
                   drot=float(a["drot"]), time_start=t0, time_end=t1)
    cast = load_seabird_cnv(str(DATA / st / CNV[st]))
    z = ctd_depth_on_ping(cast, d.time)
    r = detect_bottom_and_flag(d, z, own_bottom_track(d))
    d.weight = d.weight + np.where(np.isnan(r["izmflag"]), np.nan, 0.0)
    se = form_superensembles(d, z, zbottom=r["zbottom"], avdz=float(a["avdz"]),
                             superens_std_min=float(a["superens_std_min"]),
                             outlier_n=int(a["outlier_n"]))
    res = invert(se, uship=float(a["uship"]) + 1j * float(a["vship"]),
                 zbottom=r["zbottom"], dt_profile=float(a["dt_profile"]),
                 dz=float(a["dz"]), smoofac=float(a["smoofac"]),
                 botfac=float(a["botfac"]), barofac=float(a["barofac"]),
                 outlier=float(a["outlier"]))
    return res, ds


# --------------------------------------------------------------------------- #
# unit-level
# --------------------------------------------------------------------------- #
def test_lainsmoo_curvature_rows():
    """Uniform-weight columns yield interior [-1,2,-1] curvature rows + doubled edges."""
    rows = _lainsmoo(np.ones(5), smoofac=1.0)
    interior = [r for r in rows if r[0].size == 3]
    assert len(interior) == 3
    vals = interior[0][1]
    assert np.allclose(vals / vals[1], [-0.5, 1.0, -0.5])   # proportional to [-1,2,-1]


def test_lanarrow_rejects_worst_point():
    """The single worst-residual measurement is flagged at its (bin, ens)."""
    from ladcp.ix.inverse import _Solution
    nz, nt = 4, 3
    # one measurement with a huge residual at (bin=1, ens=2)
    d = np.array([0.0, 0.0, 5.0])
    jz = np.array([0, 1, 2])
    iens = np.array([0, 1, 2])
    ibin = np.array([0, 0, 1])
    sol = _Solution(uocean=np.zeros(nz, complex), uctd=np.zeros(nt, complex), velerr=0.1,
                    z=np.arange(nz), nz=nz, nt=nt, n_bt=0, nvel=np.zeros(nz),
                    d=d, jz=jz, iens=iens, ibin=ibin)
    rej = _lanarrow_reject(sol, (2, 3), frac=0.34)   # 1 of 3 points
    assert rej[1, 2]
    assert rej.sum() == 1


# --------------------------------------------------------------------------- #
# golden validation (quiet stations)
# --------------------------------------------------------------------------- #
@pytest.mark.parametrize("st", STATIONS)
def test_profile_structure(st):
    """Grid length within +/-2 of the golden, physical, finite ocean velocities."""
    exp = _expected()[str(int(st))]
    res, ds = _run(st)
    assert abs(res.config["nz"] - exp["n_z_bins"]) <= 2
    assert np.all(np.isfinite(res.u)) and np.all(np.isfinite(res.v))
    assert np.nanmax(np.abs(res.u)) < 1.0 and np.nanmax(np.abs(res.v)) < 1.0
    assert res.z[0] == pytest.approx(8.0) and np.allclose(np.diff(res.z), 8.0)


@pytest.mark.parametrize("st", STATIONS)
def test_barotropic_reference_matches_golden(st):
    """Depth-averaged (absolute) velocity within 1.5 cm/s of the golden — the strong result."""
    exp = _expected()[str(int(st))]
    res, ds = _run(st)
    assert abs(res.ubar - exp["ubar"]) < 0.015
    assert abs(res.vbar - exp["vbar"]) < 0.015
    # profile-mean signed difference vs the gridded golden is sub-cm/s (no reference bias)
    gu, gv, gz = ds["u"].values, ds["v"].values, ds["z"].values
    ui = np.interp(gz, res.z, res.u)
    vi = np.interp(gz, res.z, res.v)
    m = np.isfinite(gu) & np.isfinite(gv)
    assert abs(np.nanmean((ui - gu)[m])) < 0.015
    assert abs(np.nanmean((vi - gv)[m])) < 0.015


@pytest.mark.parametrize("st", STATIONS)
def test_pointwise_shape_within_tolerance(st):
    """Point-wise u/v within ~12 cm/s of the golden (SADCP omitted + loadrdi fidelity)."""
    res, ds = _run(st)
    gu, gv, gz = ds["u"].values, ds["v"].values, ds["z"].values
    ui = np.interp(gz, res.z, res.u)
    vi = np.interp(gz, res.z, res.v)
    m = np.isfinite(gu) & np.isfinite(gv)
    assert np.nanmedian(np.abs(ui - gu)[m]) < 0.12
    assert np.nanmedian(np.abs(vi - gv)[m]) < 0.12
    assert 0.01 < res.velerr < 0.20
