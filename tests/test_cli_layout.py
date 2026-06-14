"""Roadmap #3 — nested output layout + cruise exports/ via the CLI."""

from __future__ import annotations

import pathlib

import matplotlib
import pytest

matplotlib.use("Agg")

from ladcp.qa.cli import main as cli_main

ROOT = pathlib.Path(__file__).resolve().parent / "fixtures"
GOOD = ROOT / "New_golden" / "Good"
CTD80 = GOOD / "CTD" / "moria-80_clean.cnv"

pytestmark = pytest.mark.skipif(not CTD80.exists(), reason="MORIA New_golden not present")


def test_nested_station_layout(tmp_path):
    rc = cli_main(["80", "--root", str(GOOD), "--out", str(tmp_path), "--no-plots"])
    assert rc in (0, 1)
    sd = tmp_path / "stations" / "MORIA-80"
    assert (sd / "MORIA-80.lad").exists()
    assert (sd / "MORIA-80_qa.json").exists()
    assert (sd / "MORIA-80.nc").exists()              # per-station NetCDF (free dep)
    # no cruise aggregate for a single station without --cruise-export
    assert not (tmp_path / "exports").exists()


def test_cruise_exports_on_flag(tmp_path):
    # only MORIA-80 ships in fixtures; a one-station cruise aggregate exercises the wiring.
    # No --cruise here, so exports carry the default cruise_id ("LADCP").
    rc = cli_main(["80", "--root", str(GOOD), "--out", str(tmp_path),
                   "--no-plots", "--cruise-export", "--formats", "odv,nc,csv"])
    assert rc in (0, 1)
    exp = tmp_path / "exports"
    assert (exp / "LADCP_ladcp_odv.txt").exists()
    assert (exp / "LADCP_ladcp.nc").exists()
    assert (exp / "LADCP_summary.csv").exists()
    # xlsx not requested -> not written even if openpyxl present
    assert not (exp / "LADCP_ladcp.xlsx").exists()


def test_no_export_keeps_text_only(tmp_path):
    rc = cli_main(["80", "--root", str(GOOD), "--out", str(tmp_path),
                   "--no-plots", "--no-export"])
    assert rc in (0, 1)
    sd = tmp_path / "stations" / "MORIA-80"
    assert (sd / "MORIA-80.lad").exists()             # LDEO text still written
    assert not (sd / "MORIA-80.nc").exists()          # exports skipped
