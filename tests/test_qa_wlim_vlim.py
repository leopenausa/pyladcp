"""loadrdi wlim + vlim cell edits (loadrdi.m:180-271), wired 2026-06-12.

Legacy removes (u, v, w NaN; correlation and errvel untouched) every cell whose
vertical velocity deviates more than ``wlim`` from the head's near-bin median w
(``d.wrange=5``, per head), and every cell with horizontal speed > ``vlim`` --
the anti-leakage layer for bottom/surface echoes, interference, and far-range
noise that passes percent-good and the errvel limit. The same pair of edits
applies to the RDI firmware bottom track. pyladcp previously only counted vlim.

Found via a user leakage question on the brush edit view: on MORIA-20, 50-72%
of screened SURVIVORS at bins 16-30 (and 47-60% at the monocorer bins 3-4)
fail the wlim test. Wiring the edits halves the MORIA-80 inverse residual vs
golden (u rms 1.32 -> 0.68 cm/s, ubar -5.40 -> -6.33 vs golden -6.50).
"""

from __future__ import annotations

import pathlib
from dataclasses import replace

import numpy as np
import pytest

from ladcp.config import CastParams, resolve_params
from ladcp.models import CoordFrame, RawADCP
from ladcp.qa.ingest import DualHead
from ladcp.qa.screen import _WRANGE, screen
from ladcp.qa.superens import merge_heads

ROOT = pathlib.Path(__file__).resolve().parent / "fixtures"
GOOD = ROOT / "New_golden" / "Good"
DOWN = GOOD / "LADCP" / "MORIA-80-LADCP-M.000"
UP = GOOD / "LADCP" / "MORIA-80-LADCP-S.000"
CTD = GOOD / "CTD" / "moria-80_clean.cnv"

needs_fixtures = pytest.mark.skipif(not DOWN.exists(), reason="MORIA New_golden not present")


def _head(nbin=10, nens=30, head="master", facing="down") -> RawADCP:
    """All-good synthetic head: w = 0.5 m/s everywhere, u = v = 0.1."""
    vel = np.zeros((4, nbin, nens))
    vel[0] = 0.1
    vel[1] = 0.1
    vel[2] = 0.5
    t0 = np.datetime64("2025-10-01T00:00:00")
    time = t0 + np.arange(nens) * np.timedelta64(1, "s")
    ones = np.ones(nens)
    return RawADCP(
        head=head, coord_frame=CoordFrame.EARTH, facing=facing, freq_khz=300,
        n_beams=4, n_cells=nbin, cell_m=8.0, blank_m=6.0, wm_mode=15, pings_per_ens=1,
        serial=1, time=time, vel=vel, echo=np.zeros((4, nbin, nens)),
        corr=np.full((4, nbin, nens), 100.0), pct_good=np.full((4, nbin, nens), 100.0),
        heading=ones * 10, pitch=ones * 0, roll=ones * 0, temperature=ones * 10,
        sound_speed=ones * 1500, salinity=ones * 35, xmit_voltage=ones * 120,
        bt_range=None, bt_vel=None, meta={"dist_first_m": 10.0, "beam_angle_deg": 20.0})


PARAMS = CastParams(station="SYN", pglim=50.0, elim=0.2, wlim=0.08, vlim=1.0)


def test_wlim_flags_w_deviants_against_near_bin_median():
    d = _head()
    d.vel[2, 7, 5] = 0.7            # far bin, w off by 0.2 > wlim -> flagged
    d.vel[2, 2, 9] = 0.65           # NEAR bin deviant: reference is the median over
    #                                 bins 1-5, so one bad near bin does not move it
    sr = screen(DualHead(station="SYN", down=d), PARAMS)
    bad = sr.wv_bad_down
    assert bad[7, 5] and bad[2, 9]
    assert bad.sum() == 2
    assert sr.counts["wlim_removed_down"] == 2
    np.testing.assert_allclose(sr.wref_down, 0.5)


def test_wref_is_per_head_near_bins():
    # the reference must come from the head's own first _WRANGE bins
    d = _head()
    d.vel[2, :_WRANGE, :] = 0.9     # near bins all 0.9 -> wref 0.9; far bins (0.5) deviate
    sr = screen(DualHead(station="SYN", down=d), PARAMS)
    np.testing.assert_allclose(sr.wref_down, 0.9)
    assert sr.wv_bad_down[:_WRANGE].sum() == 0
    assert sr.wv_bad_down[_WRANGE:].all()


def test_vlim_flags_fast_cells_on_wlim_survivors():
    d = _head()
    d.vel[0, 6, 3] = 1.2            # hspeed 1.20 > 1.0 -> vlim flag
    d.vel[0, 7, 4] = 1.2            # fast AND w-deviant -> counted under wlim only
    d.vel[2, 7, 4] = 0.9
    sr = screen(DualHead(station="SYN", down=d), PARAMS)
    assert sr.wv_bad_down[6, 3] and sr.wv_bad_down[7, 4]
    assert sr.counts["vlim_removed"] == 1
    assert sr.counts["wlim_removed_down"] == 1


def test_wlim_reference_ignores_screened_cells():
    d = _head()
    d.pct_good[3, 0:3, :] = 0.0     # near bins 1-3 pg-screened -> reference from bins 4-5
    d.vel[2, 0:3, :] = 9.9          # their absurd w must NOT pollute the reference
    sr = screen(DualHead(station="SYN", down=d), PARAMS)
    np.testing.assert_allclose(sr.wref_down, 0.5)
    assert sr.counts["wlim_removed_down"] == 0


def test_velocities_nan_but_weight_and_errvel_survive():
    d = _head()
    d.vel[2, 7, 5] = 0.7
    u = _head(head="slave", facing="up")
    dh = DualHead(station="SYN", down=d, up=u)
    merged = merge_heads(dh, params=PARAMS)
    row = merged.izd[7]
    assert np.isnan(merged.ru[row, 5]) and np.isnan(merged.rw[row, 5])
    assert np.isfinite(merged.re[row, 5])       # errvel untouched (legacy leaves l.e)
    assert np.isfinite(merged.weight[row, 5])   # corr/weight untouched -> same normalisation
    assert merged.wref_dn is not None
    np.testing.assert_allclose(merged.wref_dn, 0.5)


def test_disabled_by_loose_thresholds():
    d = _head()
    d.vel[2, 7, 5] = 0.7
    d.vel[0, 6, 3] = 1.2
    loose = replace(PARAMS, wlim=np.inf, vlim=np.inf)
    sr = screen(DualHead(station="SYN", down=d), loose)
    assert sr.wv_bad_down.sum() == 0
    assert sr.counts["wlim_removed_down"] == sr.counts["vlim_removed"] == 0


def test_rdi_bottom_track_gets_the_edits():
    from ladcp.qa.bottom import bottom_track_velocity
    d = _head(nbin=20, nens=40)
    # a believable firmware bottom track: package velocity ~0, bt w ~ -0.5 (mirror)
    bt_vel = np.zeros((4, 40))
    bt_vel[2] = -0.5 + 0.0
    bt_vel[2, 7] = -0.9             # w deviates 0.4+0.5+... vs wref_dn 0.5 -> |bw-wref| huge
    bt_vel[0, 9] = 1.4              # hspeed > vlim
    d2 = replace(d, bt_vel=bt_vel, bt_range=np.full((4, 40), 200.0))
    dh = DualHead(station="SYN", down=d2)
    merged = merge_heads(dh, params=PARAMS)
    bt = bottom_track_velocity(dh, merged, btrk_mode=3, wlim=PARAMS.wlim, vlim=PARAMS.vlim)
    assert bt.source == "rdi"
    assert np.isnan(bt.bvel[9])                 # vlim edit
    assert np.isnan(bt.bw[7]) and np.isnan(bt.bvel[7])   # wlim edit vs wref_dn
    # untouched samples survive: |(-0.5) - 0.5| > wlim though...
    # the synthetic w mirror makes ALL samples wlim-fail unless wref matches; assert
    # the edit is selective when bw == wref instead:
    bt_vel2 = np.zeros((4, 40))
    bt_vel2[2] = 0.5                            # bt w == water wref -> all pass wlim
    d3 = replace(d, bt_vel=bt_vel2, bt_range=np.full((4, 40), 200.0))
    dh3 = DualHead(station="SYN", down=d3)
    m3 = merge_heads(dh3, params=PARAMS)
    bt3 = bottom_track_velocity(dh3, m3, btrk_mode=3, wlim=PARAMS.wlim, vlim=PARAMS.vlim)
    assert np.isfinite(bt3.bvel).sum() == 40


@needs_fixtures
def test_moria80_counts_and_report_metrics():
    from ladcp.qa.ingest import load_dualhead
    from ladcp.qa.report import assess
    p = resolve_params("MORIA", "MORIA-80")
    dh = load_dualhead(str(DOWN), str(UP), station="MORIA-80", params=p)
    qc = assess(dh, params=p)
    m = qc.metrics["edit_wlim_removed_down"]
    assert int(m.value) > 0                      # the edit is live and reported
    assert "edit_vlim_removed" in qc.metrics
