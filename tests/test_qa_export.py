"""Phase 5e/5f — .lad export and velocity figure."""

from __future__ import annotations

import pathlib

import numpy as np
import pytest

from ladcp.config import moria05_params
from ladcp.io.ctd_cnv import read_ctd_cnv
from ladcp.qa.export import write_lad
from ladcp.qa.ingest import load_dualhead
from ladcp.qa.inverse import compute_velocity

ROOT = pathlib.Path(__file__).resolve().parent / "fixtures"  # tests/fixtures/
GOOD = ROOT / "New_golden" / "Good"
DOWN = GOOD / "LADCP" / "MORIA-80-LADCP-M.000"
UP = GOOD / "LADCP" / "MORIA-80-LADCP-S.000"
CTD = GOOD / "CTD" / "moria-80_clean.cnv"

pytestmark = pytest.mark.skipif(not DOWN.exists(), reason="MORIA New_golden not present")


@pytest.fixture(scope="module")
def vp():
    dh = load_dualhead(str(DOWN), str(UP), station="MORIA-80", params=moria05_params())
    ctd = read_ctd_cnv(str(CTD), params=moria05_params())
    return compute_velocity(dh, ctd, drot=-9.878379, params=moria05_params())


def test_write_lad_roundtrip(vp, tmp_path):
    path = tmp_path / "MORIA-80.lad"
    write_lad(vp, str(path), station="MORIA-80", lat=62.16, lon=-11.53, drot=-9.878379)
    lines = path.read_text().splitlines()
    assert lines[0].startswith("Filename")
    assert "Columns     = z:u:v:ev" in lines
    assert "Deviation   = -9.878379" in lines
    # data rows: 4 numeric columns each
    data = [ln.split() for ln in lines if ln and ln[0] in " 0123456789" and "=" not in ln]
    data = [r for r in data if len(r) == 4]
    assert len(data) > 100
    z0, u0, v0, e0 = (float(x) for x in data[0])
    assert z0 == pytest.approx(vp.z[np.isfinite(vp.u)][0], abs=8.0)
    assert e0 > 0


def test_lad_skips_nan(vp, tmp_path):
    path = tmp_path / "x.lad"
    write_lad(vp, str(path), station="x")
    body = [ln for ln in path.read_text().splitlines() if "=" not in ln and ln.strip()]
    assert all("nan" not in ln.lower() for ln in body)


def test_velocity_figure_renders(vp, tmp_path):
    import matplotlib
    matplotlib.use("Agg")
    from ladcp.plots.velocity_figure import velocity_figure

    png = tmp_path / "vel.png"
    velocity_figure(vp, station="MORIA-80", savepath=str(png))
    assert png.exists() and png.stat().st_size > 5000
