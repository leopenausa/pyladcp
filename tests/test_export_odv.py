"""Roadmap #3 — ODV Generic Spreadsheet writer."""

from __future__ import annotations

import io

import pandas as pd

from ladcp.export.odv import write_odv
from ladcp.export.tables import profile_frame


def test_odv_header_and_rows(synth_exports, tmp_path):
    path = tmp_path / "MORIA_ladcp_odv.txt"
    write_odv(synth_exports, str(path), cruise="MORIA")
    text = path.read_text()
    lines = text.splitlines()

    assert lines[0].startswith("//")                       # provenance comments
    header = next(ln for ln in lines if not ln.startswith("//"))
    cols = header.split("\t")
    assert cols[:3] == ["Cruise", "Station", "Type"]
    assert "Longitude [degrees_east]" in cols
    assert "Latitude [degrees_north]" in cols
    assert cols[-4:] == ["Depth [m]", "Eastward velocity [m/s]",
                         "Northward velocity [m/s]", "Velocity error [m/s]"]

    # total data rows = sum of finite profile bins across stations
    expected = sum(len(profile_frame(e.result)) for e in synth_exports)
    data_lines = [ln for ln in lines if ln and not ln.startswith("//") and ln != header]
    assert len(data_lines) == expected


def test_odv_numeric_roundtrip(synth_exports, tmp_path):
    path = tmp_path / "odv.txt"
    write_odv(synth_exports, str(path), cruise="MORIA")
    # drop the full-line // provenance ourselves (a "/" comment char would split "[m/s]")
    body = "\n".join(ln for ln in path.read_text().splitlines() if not ln.startswith("//"))
    df = pd.read_csv(io.StringIO(body), sep="\t")
    assert "Depth [m]" in df.columns
    assert df["Eastward velocity [m/s]"].notna().all()
    # the first station's metadata appears once, then blanks on its continuation rows
    assert (df["Station"] == "MORIA-79").sum() == 1        # only the first sample row filled
