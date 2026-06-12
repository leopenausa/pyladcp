"""Monocorer near-field mask: user-opt-in only, mask plumbing, errvel-ratio detector.

A rigid device hung ~25-28 m below the package contaminated the MORIA down-looker's
26 m + 34 m cells on the operational block 03-28 (except the verified-clean 07-10).
Masking those bins (``edit_nearfield_dn_bins``) is ALWAYS the user's explicit call --
no cruise preset sets it (user decision 2026-06-12). The non-destructive near/far
|errvel| ratio detects the situation on any cruise, and its WARN note names the bins
to mask.
"""

from __future__ import annotations

import pathlib
from dataclasses import replace

import numpy as np
import pytest

from ladcp.config import CastParams, resolve_params
from ladcp.io.ctd_cnv import read_ctd_cnv
from ladcp.models import CoordFrame, RawADCP, Status
from ladcp.qa.ingest import DualHead, load_dualhead
from ladcp.qa.inverse import compute_velocity_full
from ladcp.qa.screen import NEARFIELD_WARN_RATIO, nearfield_errvel_ratio

ROOT = pathlib.Path(__file__).resolve().parent / "fixtures"
GOOD = ROOT / "New_golden" / "Good"
DOWN = GOOD / "LADCP" / "MORIA-80-LADCP-M.000"
UP = GOOD / "LADCP" / "MORIA-80-LADCP-S.000"
CTD = GOOD / "CTD" / "moria-80_clean.cnv"

needs_fixtures = pytest.mark.skipif(not DOWN.exists(), reason="MORIA New_golden not present")
DROT = -9.878379


# --------------------------------------------------------------------------- #
# config gating
# --------------------------------------------------------------------------- #
@pytest.mark.parametrize("station", [
    "MORIA-03", "MORIA-22", "MORIA-25f", "MORIA-28",     # the former auto-masked block
    "MORIA-07", "MORIA-80", "MORIA-rinse",
])
def test_no_preset_ever_sets_the_mask(station):
    """The mask is always user-opt-in: even monocorer-block stations resolve empty."""
    assert resolve_params("MORIA", station).edit_nearfield_dn_bins == ()


def test_default_is_empty():
    assert CastParams(station="X").edit_nearfield_dn_bins == ()


def test_cli_style_override_applies():
    p = resolve_params("MORIA", "MORIA-22", overrides={"edit_nearfield_dn_bins": (3, 4)})
    assert p.edit_nearfield_dn_bins == (3, 4)


# --------------------------------------------------------------------------- #
# mask plumbing into the solve
# --------------------------------------------------------------------------- #
@needs_fixtures
def test_mask_reaches_velocity_edit():
    from ladcp.qa.superens import _edit_velocity_mask, merge_heads
    p = resolve_params("MORIA", "MORIA-80")
    dh = load_dualhead(str(DOWN), str(UP), station="MORIA-80", params=p)
    merged = merge_heads(dh, params=p)
    z = np.full(merged.ru.shape[1], 500.0)
    izm = z[None, :] + merged.offset[:, None]
    mask = _edit_velocity_mask(izm, z, merged, zbottom=None, edit_sidelobes=False,
                               dzbelow=16.0, mask_dn_bins=(1, 3, 4), mask_up_bins=(1,))
    izd = merged.izd
    assert mask[izd[0]].all() and mask[izd[2]].all() and mask[izd[3]].all()
    assert not mask[izd[1]].any()                    # the clean 18 m cell is NOT masked


@needs_fixtures
def test_masked_solve_close_on_clean_cast():
    # masking 26+34 m on a clean cast must remove data, not move the answer
    p = resolve_params("MORIA", "MORIA-80")
    dh = load_dualhead(str(DOWN), str(UP), station="MORIA-80", params=p)
    ctd = read_ctd_cnv(str(CTD), params=p)
    base = compute_velocity_full(dh, ctd, drot=DROT, params=p).vp
    p_mask = resolve_params("MORIA", "MORIA-80",
                            overrides={"edit_nearfield_dn_bins": (3, 4)})
    masked = compute_velocity_full(dh, ctd, drot=DROT, params=p_mask).vp
    assert abs(masked.ubar - base.ubar) < 0.01
    assert abs(masked.vbar - base.vbar) < 0.01
    n = min(base.z.size, masked.z.size)
    m = np.isfinite(base.u[:n]) & np.isfinite(masked.u[:n])
    assert np.sqrt(np.nanmean((masked.u[:n][m] - base.u[:n][m]) ** 2)) < 0.02


# --------------------------------------------------------------------------- #
# errvel-ratio detector
# --------------------------------------------------------------------------- #
def _synthetic_head(contaminated: bool) -> RawADCP:
    rng = np.random.default_rng(7)
    nbin, nens = 16, 200
    vel = np.zeros((4, nbin, nens))
    vel[3] = rng.normal(0, 0.03, (nbin, nens))      # baseline |errvel| ~ 0.024
    if contaminated:
        vel[3, 2:4] *= 2.2                          # device band at bins 3-4 (26/34 m)
    t0 = np.datetime64("2025-10-01T00:00:00")
    time = t0 + np.arange(nens) * np.timedelta64(1, "s")
    ones = np.ones(nens)
    return RawADCP(
        head="master", coord_frame=CoordFrame.EARTH, facing="down", freq_khz=300,
        n_beams=4, n_cells=nbin, cell_m=8.0, blank_m=6.0, wm_mode=15, pings_per_ens=1,
        serial=1, time=time, vel=vel, echo=np.zeros((4, nbin, nens)),
        corr=np.full((4, nbin, nens), 100.0), pct_good=np.full((4, nbin, nens), 100.0),
        heading=ones * 10, pitch=ones * 0, roll=ones * 0, temperature=ones * 10,
        sound_speed=ones * 1500, salinity=ones * 35, xmit_voltage=ones * 120,
        bt_range=None, bt_vel=None, meta={"dist_first_m": 10.0, "beam_angle_deg": 20.0})


def test_detector_fires_on_contaminated():
    dh = DualHead(station="SYN", down=_synthetic_head(contaminated=True))
    r = nearfield_errvel_ratio(dh)
    assert r > NEARFIELD_WARN_RATIO


def test_detector_quiet_on_clean():
    dh = DualHead(station="SYN", down=_synthetic_head(contaminated=False))
    r = nearfield_errvel_ratio(dh)
    assert np.isfinite(r) and r < 1.3


@needs_fixtures
def test_metric_ok_on_moria80():
    from ladcp.qa.report import assess
    p = resolve_params("MORIA", "MORIA-80")
    dh = load_dualhead(str(DOWN), str(UP), station="MORIA-80", params=p)
    qc = assess(dh, params=p)
    m = qc.metrics["nearfield_errvel_ratio"]
    assert m.status == Status.OK
    assert float(m.value) < NEARFIELD_WARN_RATIO


def test_metric_warn_suppressed_when_masked():
    from ladcp.qa.report import assess
    head = _synthetic_head(contaminated=True)
    dh = DualHead(station="SYN", down=head,
                  params=CastParams(station="SYN", edit_nearfield_dn_bins=(3, 4)))
    qc = assess(dh, params=dh.params)
    m = qc.metrics["nearfield_errvel_ratio"]
    assert float(m.value) > NEARFIELD_WARN_RATIO     # the signal is still reported...
    assert m.status == Status.OK                     # ...but masked -> no WARN
    qc2 = assess(replace(dh, params=None), params=CastParams(station="SYN"))
    assert qc2.metrics["nearfield_errvel_ratio"].status == Status.WARN


def test_warn_note_names_the_bins_to_mask():
    """The WARN is actionable: it suggests the exact --nearfield-dn-bins value."""
    from ladcp.qa.report import assess
    dh = DualHead(station="SYN", down=_synthetic_head(contaminated=True))
    m = assess(dh, params=CastParams(station="SYN")).metrics["nearfield_errvel_ratio"]
    assert m.status == Status.WARN
    assert ("consider --nearfield-dn-bins 3,4 "
            "(elevated |errvel| at 26-34 m below the package)") in m.note


def test_elevated_bins_locate_the_device_band():
    from ladcp.qa.screen import nearfield_elevated_bins
    bad = nearfield_elevated_bins(DualHead(station="SYN", down=_synthetic_head(True)))
    assert [(b, r) for b, r in bad] == [(3, 26.0), (4, 34.0)]   # device + echo tail
    assert nearfield_elevated_bins(
        DualHead(station="SYN", down=_synthetic_head(False))) == []
