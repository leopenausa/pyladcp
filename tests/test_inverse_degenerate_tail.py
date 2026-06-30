"""Degenerate near-bottom bin masking (ladcp.qa.inverse_full._blank_degenerate_tail).

The deepest inverse grid cell can hold only 1-2 estimates right above the seabed; the
solver then leaves it unconstrained and it swings wildly (observed on MORIA2_14: u=-2.6,
v=-2.9 m/s with nvel=2, while the cast is ~0.05 m/s). The helper blanks that trailing run
without changing the grid length (the inverse profile must stay aligned with the shear).
"""
from __future__ import annotations

import numpy as np

from ladcp.qa.inverse_full import _blank_degenerate_tail


def _profile():
    # 10 well-sampled bins (nvel~15-19) then a wild trailing bin with nvel=2
    nvel = np.array([16, 18, 17, 19, 15, 18, 16, 19, 15, 2], dtype=int)
    u = np.array([-0.06, -0.06, -0.05, -0.06, -0.06, -0.06, -0.06, -0.06, -0.05, -2.599])
    v = np.array([0.0, -0.01, 0.0, 0.01, 0.01, 0.02, 0.02, 0.02, 0.04, -2.873])
    return u, v, nvel


def test_blanks_wild_undersampled_bottom_bin():
    u, v, nvel = _profile()
    n0 = u.size
    _blank_degenerate_tail(u, v, nvel)
    assert u.size == n0 and nvel.size == n0          # grid length preserved (shear alignment)
    assert np.isnan(u[-1]) and np.isnan(v[-1]) and nvel[-1] == 0   # wild bin blanked
    assert np.isfinite(u[:-1]).all() and (nvel[:-1] > 0).all()     # good bins untouched
    assert abs(np.nanmean(u)) < 0.1                  # no -2.6 dragging the mean


def test_leaves_a_well_sampled_bottom_bin_alone():
    # deepest bin well sampled -> nothing blanked (e.g. the MORIA-80 golden case)
    nvel = np.array([16, 18, 17, 19, 18], dtype=int)
    u = np.array([-0.06, -0.06, -0.05, -0.06, -0.05])
    v = np.array([0.0, -0.01, 0.0, 0.01, 0.02])
    u0, v0, n0 = u.copy(), v.copy(), nvel.copy()
    _blank_degenerate_tail(u, v, nvel)
    np.testing.assert_array_equal(u, u0)
    np.testing.assert_array_equal(v, v0)
    np.testing.assert_array_equal(nvel, n0)


def test_blanks_a_run_not_just_one_bin():
    nvel = np.array([16, 18, 17, 2, 1], dtype=int)   # two trailing under-sampled bins
    u = np.array([-0.06, -0.06, -0.05, 3.1, -4.2])
    v = np.array([0.0, -0.01, 0.0, -2.0, 5.0])
    _blank_degenerate_tail(u, v, nvel)
    assert np.isnan(u[-2:]).all() and (nvel[-2:] == 0).all()
    assert np.isfinite(u[:3]).all()                  # interior good bins kept


def test_all_empty_is_noop():
    nvel = np.zeros(4, dtype=int)
    u = np.full(4, np.nan)
    v = np.full(4, np.nan)
    _blank_degenerate_tail(u, v, nvel)               # must not raise
    assert nvel.size == 4
