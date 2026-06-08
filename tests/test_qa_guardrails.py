"""Fail-loud guard rails: coord-frame guard, shallow/start-depth WARNs, and the bottom
stack-vs-legacy cross-check flag.

These guard the audit finding that pyladcp tended to emit plausible-but-wrong output instead of
WARN/NaN: beam-coordinate data must be refused (no beam->earth transform yet), shallow casts must
be flagged, and the depth-stack seabed must announce when it disagrees with the faithful legacy
``getdpthi`` polyfit (near-bottom multiples / shallow false-lock). All synthetic, fixture-free.
"""

from __future__ import annotations

from types import SimpleNamespace

import numpy as np
import pytest

from ladcp.models import CoordFrame, Status
from ladcp.qa.bottom import BottomResult, _legacy_seabed, bottom_metric
from ladcp.qa.depth import SyncResult
from ladcp.qa.inverse import compute_velocity_full
from ladcp.qa.report import _coord_frame_metrics, _depth_sanity_metrics

# --- bottom stack-vs-legacy cross-check -------------------------------------------------

def test_legacy_seabed_clean_recovers_bed():
    """A clean per-ping hbot at a constant bed depth -> legacy polyfit returns that depth."""
    n = 400
    z = np.concatenate([np.linspace(0, 500, n // 2), np.linspace(500, 0, n - n // 2)])
    bed = 510.0                                  # 10 m below the deepest package depth
    hbot = np.where(z > 300, bed - z, np.nan)    # echo only in the near-bottom window
    zb, err = _legacy_seabed(z, hbot, maxdepth=float(z.max()))
    assert abs(zb - bed) < 2.0
    assert err < 5.0


def test_legacy_seabed_sparse_returns_nan():
    """Too few near-bottom echoes -> undetermined (NaN), not a fabricated value."""
    n = 400
    z = np.concatenate([np.linspace(0, 500, n // 2), np.linspace(500, 0, n - n // 2)])
    hbot = np.full(n, np.nan)
    hbot[n // 2] = 10.0                           # a single near-bottom echo
    zb, _ = _legacy_seabed(z, hbot, maxdepth=float(z.max()))
    assert np.isnan(zb)


def test_bottom_metric_flags_stack_legacy_disagreement():
    """|stack - legacy| > 25 m -> WARN with an explanatory note (e.g. MORIA-89 false lock)."""
    b = BottomResult(zbottom=112.0, error=2.5, hbot=np.array([np.nan]), n_valid=180,
                     zbottom_legacy=166.0, legacy_error=2.5)
    m = bottom_metric(b)
    assert m.status == Status.WARN
    assert "differ by" in (m.note or "")


def test_bottom_metric_ok_when_stack_and_legacy_agree():
    b = BottomResult(zbottom=1080.0, error=0.5, hbot=np.array([np.nan]), n_valid=600,
                     zbottom_legacy=1080.1, legacy_error=0.3)
    m = bottom_metric(b)
    assert m.status == Status.OK
    assert "differ by" not in (m.note or "")


# --- coordinate-frame guard -------------------------------------------------------------

def test_coord_frame_metrics_flag_beam_and_gate_velocity():
    dh = SimpleNamespace(down=SimpleNamespace(coord_frame=CoordFrame.BEAM), up=None)
    metrics, earth = _coord_frame_metrics(dh)
    assert earth is False
    assert metrics[0].status == Status.WARN
    assert metrics[0].value == "beam"


def test_coord_frame_metrics_ok_for_earth_dualhead():
    dh = SimpleNamespace(down=SimpleNamespace(coord_frame=CoordFrame.EARTH),
                         up=SimpleNamespace(coord_frame=CoordFrame.EARTH))
    metrics, earth = _coord_frame_metrics(dh)
    assert earth is True
    assert all(m.status == Status.OK for m in metrics)


def test_velocity_solve_refuses_beam_coordinates():
    """The velocity solve raises rather than treat beam radials as (u, v, w, e)."""
    dh = SimpleNamespace(down=SimpleNamespace(coord_frame=CoordFrame.BEAM), up=None)
    with pytest.raises(ValueError, match="beam"):
        compute_velocity_full(dh, ctd=None, solver="shear")


# --- shallow / start-depth sanity -------------------------------------------------------

def _sync(z_on_ping, maxdepth):
    return SyncResult(lag=0, corr=0.95, z_on_ping=np.asarray(z_on_ping, float),
                      maxdepth=float(maxdepth), ctd_maxdepth=float(maxdepth))


def test_depth_sanity_flags_shallow_and_clipped_start():
    sync = _sync([5., 60, 70, 80, 70, 60, 5], maxdepth=80)
    names = {m.name: m for m in _depth_sanity_metrics(sync)}
    assert names["start_depth"].status == Status.WARN      # first in-water depth 60 m > 50
    assert names["shallow_cast"].status == Status.WARN      # maxdepth 80 m < 100


def test_depth_sanity_quiet_on_normal_deep_cast():
    sync = _sync([5., 12, 200, 500, 200, 12, 5], maxdepth=500)
    names = {m.name: m for m in _depth_sanity_metrics(sync)}
    assert names["start_depth"].status == Status.OK
    assert "shallow_cast" not in names
