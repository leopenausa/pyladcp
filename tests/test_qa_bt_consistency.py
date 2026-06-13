"""Bottom-track vs solution consistency: 2-pass discard + WARN (legacy checkbtrk / STEP-12).

Ported 2026-06-13 from the shallow-cast BT investigation. A thin or biased bottom track --
typical on shallow casts, where few super-ensembles sit 50-300 m above the seabed -- can
over-pull the barotropic reference. We judge each BT super-ensemble against an independent,
BT-free reference solution and, when a *majority* disagree by more than ``bt_consistency_max``,
drop them and re-solve. Validated: FDCCC t98-01 ubar +18.2 -> +15.2 cm/s (legacy +15.3),
MORIA-80 bit-inert (3/42 inconsistent -> no discard), t99-01 untouched (48% < majority).
"""

from __future__ import annotations

import pathlib
from dataclasses import replace
from types import SimpleNamespace

import numpy as np
import pytest

from ladcp.config import moria05_params, resolve_params
from ladcp.io.ctd_cnv import read_ctd_cnv
from ladcp.models import Status
from ladcp.qa.checks import consistency_checks
from ladcp.qa.ingest import load_dualhead
from ladcp.qa.inverse import _bt_solution_anomaly, compute_velocity_full

ROOT = pathlib.Path(__file__).resolve().parent / "fixtures"
GOOD = ROOT / "New_golden" / "Good"
DOWN = GOOD / "LADCP" / "MORIA-80-LADCP-M.000"
UP = GOOD / "LADCP" / "MORIA-80-LADCP-S.000"
CTD = GOOD / "CTD" / "moria-80_clean.cnv"
DROT = -9.878379
needs_fixtures = pytest.mark.skipif(not DOWN.exists(), reason="MORIA New_golden not present")

# FDCCC1 lives outside the repo; the shallow-cast end-to-end check runs only when present.
FD = pathlib.Path("/home/leo/Cruises/FDCCC1_test")
T98_DN = FD / "CCC_ladcp_test" / "LADCP" / "MA019000.000"
T98_CNV = FD / "pyladcp_work" / "ctd_from_hex" / "t98-01_clean.cnv"
needs_fdccc = pytest.mark.skipif(not T98_DN.exists(), reason="FDCCC1 dataset not present")


def _load():
    p = moria05_params()
    dh = load_dualhead(str(DOWN), str(UP), station="MORIA-80", params=p)
    ctd = read_ctd_cnv(str(CTD), params=p)
    return dh, ctd, p


# ----------------------------------------------------------------- unit: anomaly

def _fake_se(u_oce, u_pkg, zc):
    """One super-ensemble: se.ru = u_oce(cell) - u_pkg (legacy convention)."""
    nb = len(zc)
    ru = np.array(u_oce, float)[:, None] - u_pkg.real
    rv = np.zeros((nb, 1)) - u_pkg.imag
    return SimpleNamespace(ru=ru, rv=rv, izm=np.array(zc, float)[:, None])


def test_anomaly_zero_when_bt_matches_solution():
    zc = [40.0, 48.0, 56.0]
    u_oce = [0.10, 0.12, 0.14]
    u_pkg = 0.20 + 0.05j
    se = _fake_se(u_oce, u_pkg, zc)
    aux = SimpleNamespace(bvel=np.array([-u_pkg]))          # BT measures -u_package
    z = np.array([40.0, 48.0, 56.0]); uo = np.array(u_oce); vo = np.zeros(3)
    anom = _bt_solution_anomaly(se, aux, z, uo, vo)
    assert abs(anom[0]) < 1e-9


def test_anomaly_equals_bt_offset():
    zc = [40.0, 48.0, 56.0]
    u_oce = [0.10, 0.12, 0.14]
    u_pkg = 0.20 + 0.05j
    se = _fake_se(u_oce, u_pkg, zc)
    offset = 0.07 + 0.0j
    aux = SimpleNamespace(bvel=np.array([-(u_pkg + offset)]))   # BT biased by +offset
    z = np.array(zc, float); uo = np.array(u_oce); vo = np.zeros(3)
    anom = _bt_solution_anomaly(se, aux, z, uo, vo)
    assert abs(anom[0] - offset) < 1e-9


def test_anomaly_nan_without_bt():
    se = _fake_se([0.1, 0.1], 0.2 + 0j, [40.0, 48.0])
    aux = SimpleNamespace(bvel=np.array([np.nan]))
    anom = _bt_solution_anomaly(se, aux, np.array([40.0, 48.0]), np.array([0.1, 0.1]),
                                np.zeros(2))
    assert np.isnan(anom[0])


# ----------------------------------------------------------- MORIA-80: inert control

@needs_fixtures
def test_moria80_bit_inert():
    """Well-sampled cast: the BT is consistent, so nothing is dropped and ubar is unchanged."""
    dh, ctd, p = _load()
    r_on = compute_velocity_full(dh, ctd, drot=DROT, params=p, solver="inverse")
    r_off = compute_velocity_full(dh, ctd, drot=DROT,
                                  params=replace(p, bt_consistency_max=0.0), solver="inverse")
    assert r_on.bt_consistency is not None
    _, _, n_bt, n_drop = r_on.bt_consistency
    assert n_bt > 20                                        # well-sampled
    assert n_drop == 0                                     # minority inconsistent -> no discard
    assert abs(r_on.vp.ubar - r_off.vp.ubar) < 1e-4        # bit-inert


@needs_fixtures
def test_metric_present_and_ok_on_clean_cast():
    dh, ctd, p = _load()
    r = compute_velocity_full(dh, ctd, drot=DROT, params=p, solver="inverse")
    names = {m.name: m for m in consistency_checks(r)}
    assert "bottom_track_solution_consistency" in names
    assert names["bottom_track_solution_consistency"].status == Status.OK


@needs_fixtures
def test_off_when_threshold_zero():
    dh, ctd, p = _load()
    r = compute_velocity_full(dh, ctd, drot=DROT,
                              params=replace(p, bt_consistency_max=0.0), solver="inverse")
    assert r.bt_consistency is None


@needs_fixtures
def test_none_for_shear_solver():
    dh, ctd, p = _load()
    r = compute_velocity_full(dh, ctd, drot=DROT, params=p, solver="shear")
    assert r.bt_consistency is None


# ------------------------------------------------- FDCCC t98-01: shallow-cast fix (local only)

@needs_fdccc
def test_shallow_cast_bt_discard_lands_near_legacy():
    p = resolve_params("MORIA", "t98-01")
    dh = load_dualhead(str(T98_DN), str(T98_DN.with_name("SL019000.000")),
                       station="t98-01", params=p)
    ctd = read_ctd_cnv(str(T98_CNV), params=p)
    r_on = compute_velocity_full(dh, ctd, drot=1.75, params=p, solver="inverse")
    r_off = compute_velocity_full(dh, ctd, drot=1.75,
                                  params=replace(p, bt_consistency_max=0.0), solver="inverse")
    _, _, n_bt, n_drop = r_on.bt_consistency
    assert n_drop > 0.5 * n_bt                              # systematically biased BT
    assert r_off.vp.ubar * 100 > 17.0                      # uncorrected over-pull
    assert abs(r_on.vp.ubar * 100 - 15.3) < 1.0            # lands on legacy +15.3
