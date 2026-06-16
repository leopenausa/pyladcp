"""Independent empirical velocity uncertainty (ship-ADCP withheld).

The honest LADCP velocity-error number is RMS(LADCP - ship-ADCP) computed from a solve with
the ship-ADCP *withheld* (sadcpfac=0), so the comparison is not circular. These tests inject a
synthetic ship-ADCP offset into MORIA-80 and prove that:

  * the in-sample RMS (SADCP used as a constraint, sadcpfac>0) is optimistic -- the solution is
    pulled toward the SADCP, so the discrepancy is small;
  * the withheld RMS recovers the true ~0.15 m/s offset; and
  * the per-station ``sadcp_independent_rms`` is wired through ``_velocity_outputs`` and the
    ``consistency_checks`` scorecard, and is ``None`` when no SADCP constraint was active.
"""

from __future__ import annotations

import pathlib

import numpy as np
import pytest

from ladcp.config import moria05_params
from ladcp.io.ctd_cnv import read_ctd_cnv
from ladcp.plots.sadcp_figure import sadcp_rms_discrepancy
from ladcp.qa.checks import consistency_checks
from ladcp.qa.ingest import load_dualhead
from ladcp.qa.inverse import compute_velocity_full

ROOT = pathlib.Path(__file__).resolve().parent / "fixtures"
GOOD = ROOT / "New_golden" / "Good"
DOWN = GOOD / "LADCP" / "MORIA-80-LADCP-M.000"
UP = GOOD / "LADCP" / "MORIA-80-LADCP-S.000"
CTD = GOOD / "CTD" / "moria-80_clean.cnv"

pytestmark = pytest.mark.skipif(not DOWN.exists(), reason="MORIA New_golden not present")
DROT = -9.878379
OFFSET = 0.15


@pytest.fixture(scope="module")
def station():
    p = moria05_params()
    dh = load_dualhead(str(DOWN), str(UP), station="MORIA-80", params=p)
    ctd = read_ctd_cnv(str(CTD), params=p)
    base = compute_velocity_full(dh, ctd, drot=DROT, params=p, solver="inverse").vp
    # synthetic ship-ADCP = base solution shifted east by a known OFFSET over the upper ocean
    zs = np.array([40.0, 80.0, 120.0, 160.0, 200.0])
    svel = np.column_stack([zs, np.interp(zs, base.z, base.u) + OFFSET,
                            np.interp(zs, base.z, base.v), np.full(zs.size, 0.05)])
    return dh, ctd, p, svel


def _solve(dh, ctd, p, svel, fac):
    return compute_velocity_full(dh, ctd, drot=DROT, params=p, solver="inverse",
                                 sadcp=svel, sadcpfac=fac)


def test_withheld_rms_exceeds_in_sample(station):
    # the circular (in-sample) RMS understates the error; withholding the SADCP recovers it
    dh, ctd, p, svel = station
    in_sample = sadcp_rms_discrepancy(_solve(dh, ctd, p, svel, 3.0))
    withheld = sadcp_rms_discrepancy(_solve(dh, ctd, p, svel, 0.0))
    assert withheld > in_sample
    # withheld RMS recovers the injected ~0.15 m/s offset (u shifted, v unchanged)
    assert 0.10 < withheld < 0.20
    # in-sample is pulled small by the constraint
    assert in_sample < withheld - 0.03


def test_scorecard_has_both_metrics(station):
    # _velocity_outputs sets sadcp_independent_rms; here we mimic it and check the scorecard rows
    dh, ctd, p, svel = station
    result = _solve(dh, ctd, p, svel, 3.0)
    result.sadcp_independent_rms = sadcp_rms_discrepancy(_solve(dh, ctd, p, svel, 0.0))
    names = {m.name: m for m in consistency_checks(result)}
    assert "sadcp_consistency" in names           # in-sample (circular)
    assert "sadcp_independent_rms" in names        # withheld (honest)
    assert names["sadcp_independent_rms"].value > names["sadcp_consistency"].value


def test_no_independent_metric_without_sadcp(station):
    # a solve with no SADCP constraint carries no independent RMS and emits no extra row
    dh, ctd, p, _ = station
    result = compute_velocity_full(dh, ctd, drot=DROT, params=p, solver="inverse")
    assert result.sadcp_independent_rms is None
    names = {m.name for m in consistency_checks(result)}
    assert "sadcp_independent_rms" not in names
