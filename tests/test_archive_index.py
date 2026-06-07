"""Auto-built archive index (#4a): .hex anchor, station->file matching, incremental cache.

The Seabird header parsing, the master-matching rule, and the index->discover hand-off are
exercised synthetically (fast, CI-safe). An end-to-end build over the real raw archive runs
only when raw_CTD/ + raw_ladcp_test/ are present.
"""

from __future__ import annotations

import pathlib

import numpy as np
import pytest

from ladcp import archive as A
from ladcp import discovery as D
from ladcp.io.ctd_hex import _dm_to_deg, _station_from_name, read_hex_header

_HEADER = (
    "* Sea-Bird SBE 9 Data File:\n"
    "* System UpLoad Time = Oct 03 2025 06:26:01\n"
    "* NMEA Latitude = 62 09.68 N\n"
    "* NMEA Longitude = 011 31.85 W\n"
    "* NMEA UTC (Time) = Oct 03 2025  06:25:56\n"
    "** Ship:  Sarmiento de Gamboa \n"
    "** Cruise:  MORIA \n"
    "** Station:   ST80 \n"
    "** Depth:   1086 \n"
    "*END*\n"
)

_REPO = pathlib.Path(__file__).resolve().parents[1]
_RAW_CTD = _REPO / "raw_CTD"
_RAW_LADCP = _REPO / "raw_ladcp_test" / "LADCP" / "Data"
_HAS_RAW = _RAW_CTD.exists() and (_RAW_LADCP / "MASTER").exists()


# --- .hex header reader -------------------------------------------------------------
def test_dm_to_deg_signs():
    assert _dm_to_deg("62 09.68 N") == pytest.approx(62.16133, abs=1e-4)
    assert _dm_to_deg("011 31.85 W") == pytest.approx(-11.53083, abs=1e-4)
    assert _dm_to_deg("00 30.0 S") == pytest.approx(-0.5)


def test_station_from_filename():
    assert _station_from_name(pathlib.Path("MORIA-80-CTD.hex")) == "MORIA-80"
    assert _station_from_name(pathlib.Path("FDCCC1_002_ctd.hdr")) == "FDCCC1_002"


def test_read_hex_header_parses_anchor(tmp_path):
    f = tmp_path / "MORIA-80-CTD.hex"
    f.write_text(_HEADER + "\x00\x01\x02binary-follows")
    h = read_hex_header(f)
    assert h.station == "MORIA-80" and h.raw_station == "ST80"
    assert h.utc == np.datetime64("2025-10-03T06:25:56")     # NMEA UTC preferred
    assert h.time_source == "NMEA UTC (Time)"
    assert h.lat == pytest.approx(62.1613, abs=1e-4)
    assert h.lon == pytest.approx(-11.5308, abs=1e-4)
    assert h.depth == pytest.approx(1086.0) and h.cruise == "MORIA"


def test_time_source_falls_back_when_no_nmea_utc(tmp_path):
    hdr = _HEADER.replace("* NMEA UTC (Time) = Oct 03 2025  06:25:56\n", "")
    f = tmp_path / "MORIA-80-CTD.hdr"
    f.write_text(hdr)
    h = read_hex_header(f)
    assert h.time_source == "System UpLoad Time"             # next in preference order
    assert h.utc == np.datetime64("2025-10-03T06:26:01")


# --- master matching rule -----------------------------------------------------------
def _span(a, b):
    return np.datetime64(a), np.datetime64(b)


def test_match_master_prefers_containing_window():
    spans = {
        "early": _span("2025-10-03T00:00", "2025-10-03T01:00"),
        "cast":  _span("2025-10-03T06:20", "2025-10-03T07:05"),
        "late":  _span("2025-10-03T13:00", "2025-10-03T14:00"),
    }
    rel, prov = A._match_master(np.datetime64("2025-10-03T06:25:56"), spans)
    assert rel == "cast" and prov == "ctd-utc-in-master-window"


def test_match_master_nearest_within_tolerance_then_unmatched():
    spans = {"m": _span("2025-10-03T06:40", "2025-10-03T07:30")}     # starts 14 min after utc
    utc = np.datetime64("2025-10-03T06:26:00")
    rel, prov = A._match_master(utc, spans)
    assert rel == "m" and prov == "ctd-utc-nearest-master-start"
    far = {"m": _span("2025-10-05T06:40", "2025-10-05T07:30")}       # > 2 h away
    assert A._match_master(utc, far) == (None, "unmatched")


# --- index -> discover hand-off -----------------------------------------------------
def test_discover_uses_index_then_globs_clean_ctd(tmp_path):
    (tmp_path / "CTD").mkdir()
    (tmp_path / "CTD" / "moria-80_clean.cnv").write_text("0 0\n")
    index = {"casts": {"MORIA-80": {
        "master": "/arch/MASTER/MLADC037.000", "slave": "/arch/SLAVE/SLADC037.000",
        "ctd_hex": "/raw/MORIA-80-CTD.hex", "utc": "2025-10-03T06:25:56",
        "lat": 62.16, "lon": -11.53, "depth": 1086.0, "provenance": "x"}}}
    sf = D.discover("80", root=tmp_path, cruise="MORIA", index=index)
    assert sf.down.name == "MLADC037.000" and sf.up.name == "SLADC037.000"
    assert sf.ctd.name == "moria-80_clean.cnv" and sf.label == "MORIA-80"


def test_index_station_roundtrip():
    idx = {"casts": {"MORIA-80": {
        "station": "MORIA-80", "master": "m", "slave": "s", "ctd_hex": "c",
        "utc": "2025-10-03T06:25:56", "lat": 1.0, "lon": 2.0, "depth": 3.0,
        "provenance": "p"}}}
    e = A.index_station(idx, "MORIA-80")
    assert e is not None and e.master == "m" and e.slave == "s"
    assert A.index_station(idx, "MORIA-99") is None


# --- end-to-end build over the real archive -----------------------------------------
@pytest.mark.skipif(not _HAS_RAW, reason="raw_CTD/ + raw_ladcp_test/ not present")
def test_build_index_resolves_known_casts(tmp_path):
    out = tmp_path / "idx.json"
    idx = A.build_index(_RAW_LADCP, _RAW_CTD, root=_REPO, out=out)
    casts = idx["casts"]
    assert {"MORIA-79", "MORIA-80", "MORIA-82"} <= set(casts)
    assert casts["MORIA-80"]["master"].endswith("MLADC037.000")
    assert casts["MORIA-80"]["slave"].endswith("SLADC037.000")     # time-paired
    assert all(c["provenance"] == "ctd-utc-in-master-window" for c in casts.values())
    # incremental rebuild reuses the scan cache verbatim (no files changed)
    cache0 = idx["scan_cache"]
    idx2 = A.build_index(_RAW_LADCP, _RAW_CTD, root=_REPO, out=out)
    assert idx2["scan_cache"] == cache0
