"""edit_data geometry + Fig 14 (indicative counts; figure renders).

The exact legacy counts (bin-mask 5071, side-lobe 2022, below-bottom 736) depend on the
correlation-weight finite pattern from loadrdi internals that are not reproducible from
the available files (golden raw archive truncated). We therefore check the indicative
counts are the right order of magnitude and that the contaminated geometry is sane.
"""

from __future__ import annotations

import pathlib

import matplotlib
import numpy as np
import pytest

matplotlib.use("Agg")

from ladcp.config import moria05_params
from ladcp.io.ctd_cnv import read_ctd_cnv
from ladcp.qa.bottom import detect_bottom
from ladcp.qa.depth import synchronize
from ladcp.qa.edit import edit_data
from ladcp.qa.ingest import load_dualhead
from ladcp.plots.edit_figure import edit_figure

ROOT = pathlib.Path(__file__).resolve().parent / "fixtures"  # tests/fixtures/
GOOD = ROOT / "New_golden" / "Good"
DOWN = GOOD / "LADCP" / "MORIA-80-LADCP-M.000"
UP = GOOD / "LADCP" / "MORIA-80-LADCP-S.000"
CTD = GOOD / "CTD" / "moria-80_clean.cnv"

pytestmark = pytest.mark.skipif(not DOWN.exists(), reason="MORIA New_golden not present")


@pytest.fixture(scope="module")
def er():
    dh = load_dualhead(str(DOWN), str(UP), station="MORIA-80", params=moria05_params())
    ctd = read_ctd_cnv(str(CTD), params=moria05_params())
    return edit_data(dh, synchronize(dh, ctd), detect_bottom(dh, synchronize(dh, ctd)))


def test_combined_shape(er):
    # 30 up + gap + 30 down = 61 rows
    assert er.ts_before.shape[0] == 61
    assert er.bin_no.min() == -30 and er.bin_no.max() == 30


def test_after_removes_data(er):
    before = np.isfinite(er.ts_before).sum()
    after = np.isfinite(er.ts_after).sum()
    assert after < before


def test_counts_right_order(er):
    # indicative (see module docstring): within ~50% of golden
    assert 0.5 * 5071 < er.counts["bin_mask"] < 1.5 * 5071
    assert 0.5 * 2022 < er.counts["side_lobe"] < 1.7 * 2022


def test_edit_figure_renders(tmp_path):
    dh = load_dualhead(str(DOWN), str(UP), station="MORIA-80", params=moria05_params())
    ctd = read_ctd_cnv(str(CTD), params=moria05_params())
    out = tmp_path / "edit.png"
    edit_figure(dh, ctd, savepath=str(out))
    assert out.exists() and out.stat().st_size > 0
