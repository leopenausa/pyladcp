"""Validation harness (qa/validate.py): golden-dr loader + scoring.

MORIA-80's golden .mat is a committed fixture, so its targets always load. The other
ladder stations (MORIA-10, CRUISE2_001/002) live only in the local New_golden/ tree and
are skipped when absent. score() math is exercised synthetically so it runs everywhere.
"""

from __future__ import annotations

import numpy as np
import pytest

from ladcp.qa import validate as V


def test_score_perfect_and_offset():
    a = np.array([0.0, 1.0, 2.0, 3.0])
    assert V.score(a, a).corr == pytest.approx(1.0)
    assert V.score(a, a).rms == pytest.approx(0.0)
    off = V.score(a + 0.1, a)
    assert off.bias == pytest.approx(0.1)
    assert off.rms == pytest.approx(0.1)
    assert off.n == 4


def test_score_ignores_nan_pairs():
    a = np.array([0.0, np.nan, 2.0, 3.0])
    b = np.array([0.0, 1.0, np.nan, 3.0])
    s = V.score(a, b)
    assert s.n == 2                      # only indices 0 and 3 are jointly finite


def test_score_too_few_samples_is_nan():
    s = V.score(np.array([1.0]), np.array([1.0]))
    assert np.isnan(s.corr) and s.n == 1


def test_moria80_targets_load():
    dr = V.load_dr("MORIA-80")
    assert dr.z.size == dr.u.size == dr.v.size > 100
    assert np.all(np.diff(dr.z) > 0)     # monotonic depth grid
    assert not dr.has_sadcp              # MORIA-80 ran without SADCP
    assert -0.12 < dr.ubar < 0.0         # golden ubar -0.065


def test_score_profile_against_self_is_perfect():
    dr = V.load_dr("MORIA-80")
    s = V.score_profile(dr.z, dr.u, dr.v, dr)
    assert s["u"].corr == pytest.approx(1.0)
    assert s["u"].rms == pytest.approx(0.0, abs=1e-9)


@pytest.mark.parametrize("name", ["MORIA-10", "CRUISE2_001", "CRUISE2_002"])
def test_local_only_stations(name):
    st = V.STATIONS[name]
    if not st.mat.exists():
        pytest.skip(f"{name} golden not present (local-only)")
    dr = V.load_dr(name)
    assert dr.z.size == dr.u.size > 5
    # CRUISE2_002 is the only ladder cast that truly used the SADCP constraint.
    assert dr.has_sadcp == (name == "CRUISE2_002")
    if dr.has_sadcp:
        assert np.isfinite(dr.u_sadcp).sum() > 1
