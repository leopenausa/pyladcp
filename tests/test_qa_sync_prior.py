"""Fixture-free regression for the absolute-time sync prior (MORIA-02/03/04 fix).

When the ADCP starts recording *mid-cast* (a fragmented deployment), the CTD cast begins before
the ADCP record, so the true ADCP-minus-CTD offset is NEGATIVE -- which the whole-record
W-search (:func:`_coarse_offset`) cannot represent (it can only place the CTD window *inside*
the record, offset >= 0), so it false-locks near 0. ``synchronize`` corrects this from the CTD
cast-start UTC (``ctd.meta['utc_start']``, from the archive index) only when the search disagrees
with that prior -- correctly-located casts are untouched.
"""

from __future__ import annotations

from types import SimpleNamespace

import numpy as np
import pytest

from ladcp.models import CTDTimeSeries
from ladcp.qa.depth import _score_at_offset, synchronize


def _cast(duration_s=3600, max_depth=1800.0):
    """A down-then-up CTD profile at 1 Hz; returns (t, depth, w).

    A couple of deterministic wiggles ride on the triangle so the vertical-velocity signature
    has a sharp, unique cross-correlation peak (a flat triangle alone is lag-ambiguous; real
    casts carry heave/rate structure).
    """
    t = np.arange(0.0, duration_s + 1.0)
    half = t.size // 2
    depth = np.concatenate([np.linspace(0.0, max_depth, half),
                            np.linspace(max_depth, 0.0, t.size - half)])
    depth = depth + 25.0 * np.sin(2 * np.pi * t / 237.0) + 15.0 * np.sin(2 * np.pi * t / 91.0)
    depth = np.clip(depth, 0.0, None)
    w = np.gradient(depth, t)
    return t, depth, w


def _ctd(t, depth, *, utc_start=None):
    meta = {} if utc_start is None else {"utc_start": utc_start}
    return CTDTimeSeries(time_elapsed_s=t, lat=np.full(t.size, 45.0),
                         lon=np.zeros(t.size), pressure=depth,
                         temperature=np.full(t.size, 10.0),
                         salinity=np.full(t.size, 35.0), meta=meta)


def _dh(adcp_t0: str, w_per_ping: np.ndarray):
    """Stub DualHead exposing only what synchronize touches: down.time and down.vel[2]."""
    n = w_per_ping.size
    time = np.datetime64(adcp_t0) + np.arange(n) * np.timedelta64(1, "s")
    vel = [np.zeros((1, n)), np.zeros((1, n)), w_per_ping[None, :], np.zeros((1, n))]
    return SimpleNamespace(down=SimpleNamespace(time=time, vel=vel))


def test_score_at_offset_peaks_at_true_alignment():
    t, depth, w = _cast()
    tad = t.copy()
    best = _score_at_offset(tad, w, t, w, 0.0)        # perfectly aligned
    worse = _score_at_offset(tad, w, t, w, 900.0)     # shifted 15 min
    assert best == pytest.approx(1.0, abs=1e-6)
    assert worse < best


def test_prior_rescues_midcast_adcp_fragment():
    # ADCP began 1200 s into the cast -> it only recorded cast-time [1200, 3600].
    t, depth, w = _cast(duration_s=3600, max_depth=1800.0)
    frag = w[1200:]                                   # the W the ADCP actually saw
    adcp_t0 = "2025-09-17T05:54:00"
    ctd_utc = "2025-09-17T05:34:00"                   # CTD started 1200 s (20 min) earlier
    dh = _dh(adcp_t0, frag)

    # without the prior: the search cannot reach the true -1200 s offset, so the cast is
    # mislocated (the ADCP's first ping maps to ~240 m instead of the true ~1200 m).
    bad = synchronize(dh, _ctd(t, depth))
    assert abs(bad.lag - (-1200)) > 300               # search fails to find the true offset
    assert bad.z_on_ping[0] < 600.0                   # first ping mis-mapped shallow

    # with the prior: offset snaps to ~-1200 s and the first ping maps to its true depth ~1200 m
    good = synchronize(dh, _ctd(t, depth, utc_start=ctd_utc))
    assert good.lag == pytest.approx(-1200, abs=60)
    assert good.coarse_score > 0.8                    # W signatures genuinely correlate there
    assert good.z_on_ping[0] == pytest.approx(1200.0, abs=80.0)


def test_prior_does_not_override_when_search_agrees():
    # ADCP covers the whole cast and starts with it: the search is already correct (offset ~0),
    # and a prior that agrees (within the window) must leave the result unchanged.
    t, depth, w = _cast()
    dh = _dh("2025-09-17T05:34:00", w)
    base = synchronize(dh, _ctd(t, depth))
    withp = synchronize(dh, _ctd(t, depth, utc_start="2025-09-17T05:34:00"))   # prior ~0
    assert abs(base.lag) < 60 and abs(withp.lag) < 60
    assert withp.lag == base.lag                      # agree -> search result kept verbatim
