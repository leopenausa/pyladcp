"""Phase 2 — threshold screening counts vs the MORIA-80 golden .log.

Bit-exact gates: percent-good (down/up), error-velocity (combined), and the two tilt
ensemble rejections. These are the edits that are unambiguous on raw data.
"""

from __future__ import annotations

import pathlib

import pytest

from ladcp.config import moria05_params
from ladcp.qa.golden import LOG_COUNTS
from ladcp.qa.ingest import load_dualhead
from ladcp.qa.screen import screen

ROOT = pathlib.Path(__file__).resolve().parent / "fixtures"  # tests/fixtures/
LAD = ROOT / "New_golden" / "Good" / "LADCP"
DOWN, UP = LAD / "MORIA-80-LADCP-M.000", LAD / "MORIA-80-LADCP-S.000"

pytestmark = pytest.mark.skipif(not DOWN.exists(), reason="MORIA New_golden not present")


@pytest.fixture(scope="module")
def sr():
    dh = load_dualhead(str(DOWN), str(UP), station="MORIA-80", params=moria05_params())
    return screen(dh)


def test_pg_removed_down(sr):
    assert sr.counts["pg_removed_down"] == LOG_COUNTS["pg_removed_down"]


def test_pg_removed_up(sr):
    assert sr.counts["pg_removed_up"] == LOG_COUNTS["pg_removed_up"]


def test_errvel_combined(sr):
    # golden .log: "removed 4108 values because of error velocity > 0.2 m/s"
    assert sr.counts["errvel_removed"] == 4108


def test_tilt_gt22(sr):
    assert sr.counts["tilt_removed_gt22"] == LOG_COUNTS["tilt_removed_gt22"]


def test_tilt_deriv(sr):
    assert sr.counts["tilt_removed_deriv4"] == LOG_COUNTS["tilt_removed_deriv4"]


def test_middle_hour_warning(sr):
    # golden warning fires (exact count 358 is merged-array dependent -> velocity stage);
    # here we confirm the warning is raised and the count is the right order of magnitude.
    assert any("horizontal velocities" in w and "middle hour" in w for w in sr.warnings)
    assert 100 < sr.counts["hspeed_mid_hour_approx"] < 700
