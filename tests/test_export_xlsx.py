"""Roadmap #3 — Excel workbook writers (optional openpyxl extra)."""

from __future__ import annotations

import builtins

import pandas as pd
import pytest

from ladcp.export import ExportDependencyError
from ladcp.export.tables import locate_frame, profile_frame
from ladcp.export.xlsx import write_cruise_xlsx, write_station_xlsx


def test_station_xlsx_sheets(synth_export, tmp_path):
    pytest.importorskip("openpyxl")
    path = tmp_path / "MORIA-80.xlsx"
    write_station_xlsx(synth_export, str(path))
    book = pd.read_excel(path, sheet_name=None)
    assert {"profile", "shear", "metadata", "qa", "bottom_track", "sadcp"} <= set(book)
    # data sheets are self-locating: station id + position lead each row
    assert list(book["profile"].columns[:3]) == ["station", "latitude_deg", "longitude_deg"]
    pd.testing.assert_frame_equal(
        book["profile"].reset_index(drop=True),
        locate_frame(profile_frame(synth_export.result), synth_export).reset_index(drop=True),
        check_dtype=False)


def test_cruise_xlsx_sheets(synth_exports, tmp_path):
    pytest.importorskip("openpyxl")
    path = tmp_path / "MORIA_ladcp.xlsx"
    write_cruise_xlsx(synth_exports, str(path), cruise="MORIA")
    book = pd.read_excel(path, sheet_name=None)
    assert {"profiles", "summary", "metadata"} <= set(book)
    assert set(book["profiles"]["station"]) == {"MORIA-79", "MORIA-80"}
    # stacked profiles carry per-row position
    assert {"latitude_deg", "longitude_deg"} <= set(book["profiles"].columns)
    assert len(book["summary"]) == 2


def test_missing_openpyxl_raises(synth_export, tmp_path, monkeypatch):
    real_import = builtins.__import__

    def fake_import(name, *a, **k):
        if name == "openpyxl":
            raise ImportError("no openpyxl")
        return real_import(name, *a, **k)

    monkeypatch.setattr(builtins, "__import__", fake_import)
    with pytest.raises(ExportDependencyError, match="pip install pyladcp\\[export\\]"):
        write_station_xlsx(synth_export, str(tmp_path / "x.xlsx"))
