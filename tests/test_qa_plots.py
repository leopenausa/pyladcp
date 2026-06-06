"""Phase 4 — smoke tests for the QA figures and CLI (render without error)."""

from __future__ import annotations

import pathlib

import matplotlib
import pytest

matplotlib.use("Agg")

from ladcp.config import moria05_params
from ladcp.plots.alignment import alignment_figure
from ladcp.plots.raw_dashboard import raw_dashboard
from ladcp.qa.cli import main as cli_main
from ladcp.qa.ingest import load_dualhead

ROOT = pathlib.Path(__file__).resolve().parent / "fixtures"  # tests/fixtures/
LAD = ROOT / "New_golden" / "Good" / "LADCP"
DOWN, UP = LAD / "MORIA-80-LADCP-M.000", LAD / "MORIA-80-LADCP-S.000"

pytestmark = pytest.mark.skipif(not DOWN.exists(), reason="MORIA New_golden not present")


@pytest.fixture(scope="module")
def dh():
    return load_dualhead(str(DOWN), str(UP), station="MORIA-80", params=moria05_params())


def test_raw_dashboard_renders(dh, tmp_path):
    out = tmp_path / "raw.png"
    raw_dashboard(dh, savepath=str(out))
    assert out.exists() and out.stat().st_size > 0


def test_alignment_renders(dh, tmp_path):
    out = tmp_path / "align.png"
    alignment_figure(dh, savepath=str(out))
    assert out.exists() and out.stat().st_size > 0


def test_cli_end_to_end(tmp_path):
    rc = cli_main([
        "--down", str(DOWN), "--up", str(UP),
        "--station", "MORIA-80", "--outdir", str(tmp_path),
    ])
    assert rc == 0
    assert (tmp_path / "MORIA-80_qa.txt").exists()
    assert (tmp_path / "MORIA-80_qa.json").exists()
    assert (tmp_path / "MORIA-80_raw.png").exists()
    assert (tmp_path / "MORIA-80_alignment.png").exists()
    assert (tmp_path / "MORIA-80_report.pdf").exists()


def test_cli_compact_batch(tmp_path):
    # compact form: station ids + --root auto-discovery, batch over two stations
    root = ROOT / "New_golden" / "Good"
    if not (root / "CTD" / "moria-79_clean.cnv").exists():
        pytest.skip("MORIA 79 not present")
    rc = cli_main(["79", "80", "--root", str(root), "--out", str(tmp_path)])
    assert rc in (0, 1)                       # 1 only if a station FAILs
    for st in ("MORIA-79", "MORIA-80"):
        assert (tmp_path / f"{st}_report.pdf").exists()
        assert (tmp_path / f"{st}_depth.png").exists()   # CTD auto-discovered


def test_pdf_report_with_ctd(tmp_path):
    from ladcp.io.ctd_cnv import read_ctd_cnv
    from ladcp.plots.pdf_report import build_report
    from ladcp.qa.report import assess
    ctd_path = ROOT / "New_golden" / "Good" / "CTD" / "moria-80_clean.cnv"
    dh = load_dualhead(str(DOWN), str(UP), station="MORIA-80", params=moria05_params())
    ctd = read_ctd_cnv(str(ctd_path), params=moria05_params())
    qc = assess(dh, ctd=ctd)
    paths = build_report(dh, qc, str(tmp_path), "MORIA-80", ctd=ctd)
    for key in ("MORIA-80_raw.png", "MORIA-80_alignment.png",
                "MORIA-80_depth.png", "MORIA-80_edit.png", "report.pdf"):
        assert pathlib.Path(paths[key]).exists()
    assert pathlib.Path(paths["report.pdf"]).stat().st_size > 0


def test_velocity_figures_in_report(tmp_path):
    # Phase 5e/5f: velocity, shear and inversion-diagnostics pages fold into the report
    from ladcp.io.ctd_cnv import read_ctd_cnv
    from ladcp.plots.pdf_report import build_report
    from ladcp.qa.inverse import compute_velocity_full
    from ladcp.qa.report import assess
    ctd_path = ROOT / "New_golden" / "Good" / "CTD" / "moria-80_clean.cnv"
    dh = load_dualhead(str(DOWN), str(UP), station="MORIA-80", params=moria05_params())
    ctd = read_ctd_cnv(str(ctd_path), params=moria05_params())
    qc = assess(dh, ctd=ctd)
    result = compute_velocity_full(dh, ctd, drot=-9.878379, params=moria05_params())
    assert result.bp is not None and result.bp.n_bins > 20
    assert result.shear.z.size == result.vp.z.size
    assert result.resid_z.size > 100 and (result.resid_z > 0).all()
    paths = build_report(dh, qc, str(tmp_path), "MORIA-80", ctd=ctd, velocity=result)
    for key in ("MORIA-80_velocity.png", "MORIA-80_shear.png", "MORIA-80_inverse.png"):
        assert pathlib.Path(paths[key]).exists() and pathlib.Path(paths[key]).stat().st_size > 0
