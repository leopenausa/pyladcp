"""Synthetic, fixture-free tests for the depth-stacking seabed estimator + metric.

These guard the deep-bias fix: the seabed depth is recovered by stacking raw echo amplitude
along constant-depth loci (:func:`_stack_seabed`), which rejects the transmission-loss tail
(constant *range*, not depth) and mid-water scattering layers, and falls back to the deepest
package depth when no echo stacks. The per-ping pick (:func:`bottom_distance`) is the strongest
``targ`` return (legacy ``getbtrack``) and is exercised on real data elsewhere.
"""

from __future__ import annotations

import numpy as np
import pytest

from ladcp.models import Status
from ladcp.qa.bottom import (
    _BOTTOM_ERR_WARN,
    _SEABED_STACK_MIN,
    _STACK_COVER_MIN,
    BottomResult,
    _stack_seabed,
    _support_cover,
    bottom_metric,
)


def _triangle(zmax: float, n: int = 600) -> np.ndarray:
    half = n // 2
    return np.concatenate([np.linspace(0.0, zmax, half),
                           np.linspace(zmax, 0.0, n - half)])


def _echo_field(z, zd, *, beds=(), const_range=(), bg=60.0, amp=40.0, width=6.0):
    """Build a synthetic (n_cell, n_ping) amplitude field.

    ``beds`` are constant-*depth* reflectors (each adds a bump at range = D - z per ping, so
    it stacks at depth D); ``const_range`` are fixed-*range* features (a reverberation tail,
    same range every ping, so they do NOT stack to a constant depth).
    """
    nb, npg = zd.size, z.size
    ea = np.full((nb, npg), bg) + np.random.default_rng(1).normal(0, 1.0, (nb, npg))
    for j in range(npg):
        for D in beds:
            r = D - z[j]
            ea[:, j] += amp * np.exp(-0.5 * ((zd - r) / width) ** 2)
        for r0 in const_range:
            ea[:, j] += amp * np.exp(-0.5 * ((zd - r0) / width) ** 2)
    return ea


def _zd(n=60, cell=4.0, first=10.0):
    return first + np.arange(n) * cell


def test_stack_recovers_seabed():
    z = _triangle(1000.0)
    zd = _zd()
    seabed = 1010.0
    ea = _echo_field(z, zd, beds=(seabed,))
    zb, is_echo = _stack_seabed(z, ea, zd, np.ones_like(z))
    assert is_echo
    assert zb == pytest.approx(seabed, abs=3.0)


def test_stack_rejects_constant_range_tail():
    # a strong feature at fixed *range* (reverberation/TL tail) must NOT be read as a seabed:
    # it does not stack to any constant depth, so no echo qualifies -> fallback to zmax.
    z = _triangle(1000.0)
    zd = _zd()
    ea = _echo_field(z, zd, beds=(), const_range=(150.0,))
    zb, is_echo = _stack_seabed(z, ea, zd, np.ones_like(z))
    assert not is_echo
    assert zb == pytest.approx(np.nanmax(z), abs=1.0)      # deepest package depth


def test_stack_picks_deepest_of_scatterer_and_bed():
    # a strong mid-water scattering layer above a (real) deeper seabed -> pick the seabed.
    z = _triangle(1000.0)
    zd = _zd()
    seabed = 1010.0
    ea = _echo_field(z, zd, beds=(880.0, seabed))          # layer at 880 m, bed at 1010 m
    zb, is_echo = _stack_seabed(z, ea, zd, np.ones_like(z))
    assert is_echo
    assert zb == pytest.approx(seabed, abs=4.0)


def test_stack_fallback_when_no_return():
    z = _triangle(4000.0)
    zd = _zd()
    ea = _echo_field(z, zd, beds=(), amp=0.0)              # pure background, no reflector
    zb, is_echo = _stack_seabed(z, ea, zd, np.ones_like(z))
    assert not is_echo
    assert zb == pytest.approx(np.nanmax(z), abs=1.0)


def test_support_cover_wide_vs_narrow():
    # a genuine bed at 1000 m (package reaching 1000 m) is seen across a wide span of package
    # depths -> high cover; a constant-range artifact's picks cluster in a thin band -> low cover.
    wide = _support_cover(np.linspace(750.0, 995.0, 50), zbottom=1000.0, zmax=1000.0)
    narrow = _support_cover(np.linspace(200.0, 235.0, 50), zbottom=1000.0, zmax=1000.0)
    assert wide >= 0.6 and wide >= _STACK_COVER_MIN       # genuine bed clearly accepted
    assert narrow < _STACK_COVER_MIN                      # narrow artifact band rejected
    # degenerate / empty support never divides by zero
    assert _support_cover(np.array([]), 1000.0, 1000.0) == 0.0
    # no geometrically visible span (zbottom beyond reach of zmax) -> 1.0, never rejects
    assert _support_cover(np.array([10.0, 12.0]), zbottom=40.0, zmax=10.0) == 1.0


def test_metric_ok_warn_nan():
    ok = bottom_metric(BottomResult(zbottom=1080.0, error=0.3, hbot=np.array([]), n_valid=500))
    assert ok.status is Status.OK and ok.threshold == _BOTTOM_ERR_WARN

    noisy = bottom_metric(BottomResult(zbottom=4719.0, error=25.9,
                                       hbot=np.array([]), n_valid=600))
    assert noisy.status is Status.WARN          # finite but uncertain -> flagged honestly

    # near-touch zmax fallback: finite zbottom + small error, but flagged because it is a
    # lower bound, not a stacked echo lock
    fallback = bottom_metric(BottomResult(zbottom=4718.0, error=8.0, hbot=np.array([]),
                                          n_valid=12, is_fallback=True))
    assert fallback.status is Status.WARN
    assert "lower bound" in fallback.note

    # no bottom echo at all -> NaN zbottom, flagged, distinct note
    nobed = bottom_metric(BottomResult(zbottom=np.nan, error=np.nan, hbot=np.array([]),
                                       n_valid=0, is_fallback=True))
    assert nobed.status is Status.WARN
    assert "no bottom" in nobed.note

    assert _SEABED_STACK_MIN > 0                  # sanity: threshold is configured
