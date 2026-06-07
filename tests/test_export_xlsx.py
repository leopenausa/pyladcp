"""Roadmap #3 — Excel workbook writers (optional openpyxl extra)."""

from __future__ import annotations

import builtins

import pandas as pd
import pytest

from ladcp.export import ExportDependencyError
from ladcp.export.tables import profile_frame
from ladcp.export.xlsx import write_cruise_xlsx, write_station_xlsx


def test_station_xlsx_sheets(synth_export, tmp_path):
    pytest.importorskip("openpyxl")
    path = tmp_path / "MORIA-80.xlsx"
    write_station_xlsx(synth_export, str(path))
    book = pd.read_excel(path, sheet_name=None)
    assert {"profile", "shear", "metadata", "qa", "bottom_track", "sadcp"} <= set(book)
    # profile sheet matches the source frame
    pd.testing.assert_frame_equal(
        book["profile"].reset_index(drop=True),
        profile_frame(synth_export.result).reset_index(drop=True),
        check_dtype=False)


def test_cruise_xlsx_sheets(synth_exports, tmp_path):
    pytest.importorskip("openpyxl")
    path = tmp_path / "MORIA_ladcp.xlsx"
    write_cruise_xlsx(synth_exports, str(path), cruise="MORIA")
    book = pd.read_excel(path, sheet_name=None)
    assert {"profiles", "summary", "metadata"} <= set(book)
    assert set(book["profiles"]["station"]) == {"MORIA-79", "MORIA-80"}
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
