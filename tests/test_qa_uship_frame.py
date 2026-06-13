"""GPS uship is rotated true->magnetic for the barotropic (lainbaro) constraint.

The inverse solve runs in the MAGNETIC frame (se.ru/rv are rotated to true only at the end),
but ``aux.uship`` is built in the TRUE (geographic) frame from the CTD nav. ``invert`` must
rotate it true->magnetic before the lainbaro RHS -- the same handling the SADCP constraint
gets -- so the GPS row is frame-consistent. Legacy rotates all data to true at load, so its
lainbaro uship is already consistent; pyladcp must rotate here (s03 audit, DIVERGED/high).
"""

from __future__ import annotations

import pathlib

import numpy as np
import pytest

from ladcp.config import moria05_params
from ladcp.io.ctd_cnv import read_ctd_cnv
from ladcp.qa.ingest import load_dualhead
from ladcp.qa.inverse import _ping_dt, build_solve_context
from ladcp.qa.inverse_full import inverse_inputs, invert
from ladcp.qa.superens import _uvrot

ROOT = pathlib.Path(__file__).resolve().parent / "fixtures"
GOOD = ROOT / "New_golden" / "Good"
DOWN = GOOD / "LADCP" / "MORIA-80-LADCP-M.000"
UP = GOOD / "LADCP" / "MORIA-80-LADCP-S.000"
CTD = GOOD / "CTD" / "moria-80_clean.cnv"

pytestmark = pytest.mark.skipif(not DOWN.exists(), reason="MORIA New_golden not present")
DROT = -9.878379


@pytest.fixture(scope="module")
def context():
    p = moria05_params()
    dh = load_dualhead(str(DOWN), str(UP), station="MORIA-80", params=p)
    ctd = read_ctd_cnv(str(CTD), params=p)
    se, z, merged, bt, sync, bottom = build_solve_context(dh, ctd, dz=8.0, params=p)
    aux = inverse_inputs(se, bt=bt, sync=sync, ctd=ctd, ping_dt=_ping_dt(dh),
                         zbottom=bottom.zbottom)
    return se, aux, bottom.zbottom


def _solve_uship(context, drot):
    se, aux, zbottom = context
    diag: dict = {}
    invert(se, aux, dz=8.0, drot=drot, zbottom=zbottom, botfac=1.0, barofac=1.0,
           smoofac=1.0, diag=diag)
    return diag["uship"]


def test_uship_rotated_true_to_magnetic(context):
    _, aux, _ = context
    uship_true = aux.uship
    assert np.isfinite(uship_true)            # MORIA-80 nav yields a finite (small) uship
    used = _solve_uship(context, DROT)
    su, sv = _uvrot(uship_true.real, uship_true.imag, -DROT)   # true -> magnetic
    expected = complex(su, sv)
    assert used == pytest.approx(expected, abs=1e-12)
    # a pure rotation preserves magnitude but changes the vector at this declination
    assert abs(used) == pytest.approx(abs(uship_true), rel=1e-12)
    assert abs(used - uship_true) > 0          # drot != 0 -> the constraint actually moved


def test_uship_unrotated_at_zero_declination(context):
    _, aux, _ = context
    used = _solve_uship(context, 0.0)
    assert used == pytest.approx(aux.uship, abs=1e-12)    # drot == 0 -> identity
