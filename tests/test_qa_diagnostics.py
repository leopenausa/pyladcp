"""Bottom-track diagnostics + checkinv consistency metrics + Figure 13."""

from __future__ import annotations

import types

import numpy as np

from ladcp.models import Status
from ladcp.qa.bottom import BottomTrack, BtrkDiagnostics, btrk_diagnostics
from ladcp.qa.checks import consistency_checks
from ladcp.qa.inverse import VelocityProfile, VelocityResult


def _bottom_track(n=200, u=0.05, v=0.02):
    rng = np.random.default_rng(0)
    bvel = (u + rng.normal(0, 0.02, n)) + 1j * (v + rng.normal(0, 0.02, n))
    bvel[::5] = np.nan                                   # some pings drop out
    bw = rng.normal(0.0, 0.05, n)
    hbot = 100.0 + rng.normal(0, 20.0, n)               # roughness ~20 m
    return BottomTrack(bvel=bvel, bw=bw, hbot=hbot, n_valid=int(np.isfinite(bvel).sum()))


def _dh_with_rdi(n=200, u=0.05, v=0.02):
    rng = np.random.default_rng(1)
    bt = np.full((4, n), np.nan)
    bt[0] = u + rng.normal(0, 0.02, n)
    bt[1] = v + rng.normal(0, 0.02, n)
    bt[2] = rng.normal(0, 0.05, n)
    return types.SimpleNamespace(down=types.SimpleNamespace(bt_vel=bt))


def test_btrk_diagnostics_bias_and_roughness():
    bt = _bottom_track(u=0.05, v=0.02)
    dh = _dh_with_rdi(u=0.05, v=0.02)
    d = btrk_diagnostics(bt, dh)
    assert isinstance(d, BtrkDiagnostics)
    assert d.n_own == int(np.isfinite(bt.bvel).sum()) and d.n_rdi == 200
    # own and RDI drawn from the same mean -> small inter-method bias
    assert abs(d.u_bias) < 0.02 and abs(d.v_bias) < 0.02
    assert 12 < d.roughness < 30                          # ~20 m by construction


def test_btrk_diagnostics_without_rdi():
    bt = _bottom_track()
    dh = types.SimpleNamespace(down=types.SimpleNamespace(bt_vel=None))
    d = btrk_diagnostics(bt, dh)
    assert d.n_rdi == 0 and np.isnan(d.u_bias)
    assert d.n_own > 0                                    # own track still summarised


def _result(uerr=0.03, btrk=None, sadcp=None):
    z = np.arange(20.0, 200.0, 10.0)
    vp = VelocityProfile(z=z, u=np.full_like(z, 0.1), v=np.full_like(z, -0.05),
                         uerr=np.full_like(z, uerr), nvel=np.full_like(z, 5),
                         ubar=0.1, vbar=-0.05, n=np.full_like(z, 5))
    rng = np.random.default_rng(2)
    ru = rng.normal(0, 0.06, 300)                         # noise floor ~0.06
    return VelocityResult(vp=vp, bp=None, shear=None, zbottom=300.0,
                          resid_z=z[:1], resid_u=ru, resid_v=ru, sadcp=sadcp, btrk=btrk)


def test_consistency_checks_metrics():
    bt = _bottom_track()
    dh = _dh_with_rdi()
    d = btrk_diagnostics(bt, dh)
    z = np.arange(20.0, 200.0, 10.0)
    sadcp = np.column_stack([z, np.full_like(z, 0.1), np.full_like(z, -0.05),
                             np.full_like(z, 0.1)])       # exactly matches vp -> rms 0
    r = _result(uerr=0.03, btrk=d, sadcp=sadcp)
    names = {m.name: m for m in consistency_checks(r)}
    assert {"velocity_error_vs_noise", "bottom_track_consistency",
            "sadcp_consistency"} <= set(names)
    assert names["bottom_track_consistency"].status is Status.OK
    assert names["sadcp_consistency"].value == 0.0       # SADCP == LADCP here


def test_consistency_flags_large_bt_bias():
    bt = _bottom_track(u=0.05, v=0.02)
    dh = _dh_with_rdi(u=0.35, v=0.02)                     # RDI offset 0.30 from own
    d = btrk_diagnostics(bt, dh)
    r = _result(btrk=d)
    m = {x.name: x for x in consistency_checks(r)}["bottom_track_consistency"]
    assert m.status is Status.WARN and m.value > 0.1


def test_btrack_figure_renders():
    import matplotlib
    matplotlib.use("Agg")
    from ladcp.plots.btrack_figure import btrack_figure

    d = btrk_diagnostics(_bottom_track(), _dh_with_rdi())
    assert btrack_figure(_result(btrk=d), station="T") is not None
    # no bottom track -> placeholder, no error
    assert btrack_figure(_result(btrk=None)) is not None
