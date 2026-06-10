"""Inverse constraint knobs (botfac), Figure-12 weights, down-only solve, battery cap.

Covers the tunable-constraint exposure: ``botfac`` reaching the bottom-track block,
the :class:`ConstraintWeights` diagnostics carried on the inverse result (and their
figure), the down-looker-only solve path, and the battery metric never FAILing.
"""

from __future__ import annotations

import pathlib

import matplotlib
import numpy as np
import pytest

from ladcp.config import moria05_params
from ladcp.io.ctd_cnv import read_ctd_cnv
from ladcp.qa.ingest import load_dualhead
from ladcp.qa.inverse import compute_velocity_full

matplotlib.use("Agg")

ROOT = pathlib.Path(__file__).resolve().parent / "fixtures"
GOOD = ROOT / "New_golden" / "Good"
DOWN = GOOD / "LADCP" / "MORIA-80-LADCP-M.000"
UP = GOOD / "LADCP" / "MORIA-80-LADCP-S.000"
CTD = GOOD / "CTD" / "moria-80_clean.cnv"

pytestmark = pytest.mark.skipif(not DOWN.exists(), reason="MORIA New_golden not present")
DROT = -9.878379


@pytest.fixture(scope="module")
def station():
    p = moria05_params()
    dh = load_dualhead(str(DOWN), str(UP), station="MORIA-80", params=p)
    ctd = read_ctd_cnv(str(CTD), params=p)
    return dh, ctd, p


@pytest.fixture(scope="module")
def inverse_result(station):
    dh, ctd, p = station
    return compute_velocity_full(dh, ctd, drot=DROT, params=p, solver="inverse")


def test_inverse_is_default_solver(station, inverse_result):
    dh, ctd, p = station
    r = compute_velocity_full(dh, ctd, drot=DROT, params=p)   # no solver argument
    assert np.allclose(r.vp.u, inverse_result.vp.u, equal_nan=True)


def test_weights_on_inverse_result(inverse_result):
    w = inverse_result.weights
    assert w is not None
    assert "velocity" in w.ocean and "velocity" in w.ctd
    assert "bottom track" in w.ctd                       # MORIA-80 has bottom track
    assert "GPS navigation" in w.ctd
    nz, nt = w.z.size, w.ctd["velocity"].size
    for sums, n in ((w.ocean, nz), (w.ctd, nt)):
        for vals in sums.values():
            assert vals.shape == (n,)
            assert np.all(vals >= 0)
    # data rows must dominate the solve overall
    assert w.ocean["velocity"].sum() > w.ocean.get("smoothing", np.zeros(1)).sum()


def test_weights_absent_on_shear(station):
    dh, ctd, p = station
    r = compute_velocity_full(dh, ctd, drot=DROT, params=p, solver="shear")
    assert r.weights is None


def test_botfac_zero_drops_bottom_track(station, inverse_result):
    dh, ctd, p = station
    r0 = compute_velocity_full(dh, ctd, drot=DROT, params=p, solver="inverse", botfac=0.0)
    assert "bottom track" not in r0.weights.ctd          # block never added
    # without the bottom track the barotropic reference must move
    assert not np.isclose(r0.vp.ubar, inverse_result.vp.ubar, atol=1e-4) or \
        not np.isclose(r0.vp.vbar, inverse_result.vp.vbar, atol=1e-4)


def test_botfac_scales_weights(station, inverse_result):
    # exact 3x holds only within one solve; across the two-pass driver velerr and the
    # lanarrow mask shift with botfac, so check the accumulated weight scales ~3x.
    dh, ctd, p = station
    r3 = compute_velocity_full(dh, ctd, drot=DROT, params=p, solver="inverse", botfac=3.0)
    ratio = (r3.weights.ctd["bottom track"].sum()
             / inverse_result.weights.ctd["bottom track"].sum())
    assert 2.5 < ratio < 3.5


def test_constraint_weights_figure_builds(tmp_path, inverse_result):
    from ladcp.plots.inverse_figure import constraint_weights_figure
    out = tmp_path / "w.png"
    fig = constraint_weights_figure(inverse_result.weights, station="MORIA-80",
                                    savepath=str(out))
    assert out.exists()
    import matplotlib.pyplot as plt
    plt.close(fig)


# --------------------------------------------------------------------------- #
# down-looker-only solve
# --------------------------------------------------------------------------- #
def test_down_only_solve(station, inverse_result):
    from dataclasses import replace
    dh, ctd, p = station
    r = compute_velocity_full(replace(dh, up=None), ctd, drot=DROT, params=p,
                              solver="inverse")
    vp, ref = r.vp, inverse_result.vp
    assert np.isfinite(vp.u).sum() > 20                  # a real profile, not a stub
    n = min(vp.z.size, ref.z.size)
    m = np.isfinite(vp.u[:n]) & np.isfinite(ref.u[:n])
    assert m.sum() > 20
    corr = np.corrcoef(vp.u[:n][m], ref.u[:n][m])[0, 1]
    assert corr > 0.95                                   # same ocean from half the data
    assert abs(vp.ubar - ref.ubar) < 0.05                # reference pinned the same way


def test_merge_heads_single(station):
    from dataclasses import replace

    from ladcp.qa.superens import merge_heads
    dh, _, p = station
    m = merge_heads(replace(dh, up=None), params=p)
    assert m.izu.size == 0
    assert m.ru.shape[0] == dh.down.n_cells
    assert np.all(m.offset > 0)                          # all bins below the package
    assert np.all(m.hrot == 0)


# --------------------------------------------------------------------------- #
# battery metric can WARN but never FAIL
# --------------------------------------------------------------------------- #
def test_battery_never_fails():
    from ladcp.models import Status
    from ladcp.qa.attitude import AttitudeSummary, attitude_metrics

    def summary(batt):
        return AttitudeSummary(tilt_max=5.0, tilt_mean=2.0, tilt_pct_over=0.0,
                               heading_span=360.0, xmv_median=batt / 0.33,
                               battery_v=batt, dual_head_offset_est=None,
                               pitch_offset_est=None, roll_offset_est=None)

    by_status = {batt: attitude_metrics(summary(batt)) for batt in (45.0, 38.0, 29.0)}
    get = lambda ms: next(m for m in ms if m.name == "battery").status  # noqa: E731
    assert get(by_status[45.0]) == Status.OK
    assert get(by_status[38.0]) == Status.WARN
    assert get(by_status[29.0]) == Status.WARN           # deep-low is still only WARN
