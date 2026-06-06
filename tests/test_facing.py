"""Orientation (facing) detection: precedence and the RB1606 cross-instrument case.

Background: relying on the PD0 sysconfig bit-7 alone misreads orientation on RDI
BroadBand 150 instruments (GO-SHIP RB1606), which report bit-7=1 on both heads.
Legacy loadrdi.m trusts the file's role and uses the bit only to warn; we mirror
that with precedence: explicit hint > filename heuristic > sysconfig bit.
"""

from pathlib import Path

import pytest

from ladcp.io.pd0 import _facing_from_name, read_pd0

ROOT = Path(__file__).resolve().parents[1]
RB1606 = ROOT / "test_data/goship_rb1606/008"
HAVE_RB1606 = (RB1606 / "008DL000.000").exists()


def test_facing_from_name():
    # RB1606-style embedded head tokens
    assert _facing_from_name("/x/008DL000.000") == "down"
    assert _facing_from_name("/x/008UL000.000") == "up"
    # MASTER/SLAVE parent directories
    assert _facing_from_name("/x/MASTER/foo.000") == "down"
    assert _facing_from_name("/x/SLAVE/foo.000") == "up"
    # no clear signal -> None (caller falls back to hint or sysconfig bit)
    assert _facing_from_name("/x/MLADC007.000") is None     # MORIA basename: no token
    assert _facing_from_name("/x/ambiguous.000") is None
    # conservative: incidental substrings must NOT trigger a false orientation
    assert _facing_from_name("/x/result.000") is None       # contains 'ul'
    assert _facing_from_name("/x/group_07.000") is None      # contains 'up'


@pytest.mark.skipif(not HAVE_RB1606, reason="RB1606 anchor data not present")
def test_rb1606_downlooker_resolves_down_despite_bit():
    # 008DL is documented Orientation: DOWN, but sysconfig bit-7 is set (=up).
    # Filename heuristic must win, and the disagreement must be flagged.
    r = read_pd0(str(RB1606 / "008DL000.000"))
    assert r.facing == "down"
    assert r.meta["facing_from_bit"] == "up"
    assert r.meta["facing_warning"] is not None
    assert r.freq_khz == 150           # cross-instrument: not MORIA's 300 kHz
    assert r.n_cells == 40


@pytest.mark.skipif(not HAVE_RB1606, reason="RB1606 anchor data not present")
def test_rb1606_explicit_hint_overrides():
    r = read_pd0(str(RB1606 / "008DL000.000"), facing_hint="down")
    assert r.facing == "down"
