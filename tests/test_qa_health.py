"""Phase 3 — core-health metrics vs MORIA-80 golden (beams, range, attitude)."""

from __future__ import annotations

import pathlib

import pytest

from ladcp.config import moria05_params
from ladcp.qa.beams import beam_health
from ladcp.qa.golden import load_p_struct
from ladcp.qa.ingest import load_dualhead
from ladcp.qa.range import profiling_range
from ladcp.qa.report import assess, text_report

ROOT = pathlib.Path(__file__).resolve().parent / "fixtures"  # tests/fixtures/
LAD = ROOT / "New_golden" / "Good" / "LADCP"
DOWN, UP = LAD / "MORIA-80-LADCP-M.000", LAD / "MORIA-80-LADCP-S.000"
MAT = ROOT / "New_golden" / "MORIA-80_LEO" / "MORIA-80.mat"

pytestmark = pytest.mark.skipif(not DOWN.exists(), reason="MORIA New_golden not present")


@pytest.fixture(scope="module")
def dh():
    return load_dualhead(str(DOWN), str(UP), station="MORIA-80", params=moria05_params())


@pytest.fixture(scope="module")
def golden():
    return load_p_struct(str(MAT))


def test_beam_pct_down(dh):
    # golden Figure 2 (bottom panel): [98, 95, 100, 96]
    assert beam_health(dh.down).pct.tolist() == [98, 95, 100, 96]


def test_beam_pct_up(dh):
    # golden Figure 2 (top panel): [100, 98, 100, 98]
    assert beam_health(dh.up).pct.tolist() == [100, 98, 100, 98]


def test_range_down(dh, golden):
    rr = profiling_range(dh.down)
    assert rr.beam_range.tolist() == pytest.approx(golden["dn_range"], abs=0.02)
    assert all(f == "ok" for f in rr.flags)


def test_range_up_flags_subnominal(dh, golden):
    rr = profiling_range(dh.up)
    assert rr.beam_range.tolist() == pytest.approx(golden["up_range"], abs=0.02)
    # up beams 2 & 3 are short -> must be flagged sub-nominal
    assert rr.flags[1] == "sub-nominal"
    assert rr.flags[2] == "sub-nominal"
    assert rr.status.value == "warn"


def test_battery_close(dh, golden):
    qc = assess(dh)
    bat = qc.metrics["battery"].value
    # golden battery 41.26 V (in-cast mean); median-based estimate within ~1.5 V
    assert bat == pytest.approx(golden["battery"], abs=1.5)


def test_dual_head_offset_estimate(dh, golden):
    qc = assess(dh)
    off = qc.metrics["dual_head_offset_est"].value
    # raw-ping estimate near the golden tilt-based offset (-59.57); loose by design
    assert off == pytest.approx(-59.57, abs=3.0)


def test_report_renders(dh):
    txt = text_report(assess(dh))
    assert "acquisition QA" in txt and "MORIA-80" in txt
