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

pytestmark = pytest.mark.slow

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
    assert _station_from_name(pathlib.Path("CRUISE2_002_ctd.hdr")) == "CRUISE2_002"


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
def _span(a, b, n=9999):
    return np.datetime64(a), np.datetime64(b), n


def test_match_master_prefers_containing_window():
    spans = {
        "early": _span("2025-10-03T00:00", "2025-10-03T01:00"),
        "cast":  _span("2025-10-03T06:20", "2025-10-03T07:05"),
        "late":  _span("2025-10-03T13:00", "2025-10-03T14:00"),
    }
    rel, prov = A._match_master(np.datetime64("2025-10-03T06:25:56"), spans)
    assert rel == "cast" and prov == "ctd-utc-in-master-window"


def test_match_master_picks_cast_over_stub_starting_nearer():
    # the MORIA-02/03/04 pattern: a sub-minute stub starts right at the on-station time,
    # the real cast starts later -- size must win over proximity.
    utc = np.datetime64("2025-09-17T05:17:00")
    spans = {
        "stub": _span("2025-09-17T05:11", "2025-09-17T05:12", n=29),     # starts 6 min after
        "cast": _span("2025-09-17T05:54", "2025-09-17T08:33", n=5381),   # starts 37 min after
        "tail": _span("2025-09-17T08:38", "2025-09-17T08:46", n=261),    # later stub
    }
    rel, prov = A._match_master(utc, spans)
    assert rel == "cast" and prov == "ctd-utc-nearest-cast-start"


def test_match_master_earliest_cast_wins_on_back_to_back_stations():
    # the CRUISE2 t3-01..03 pattern: consecutive ~1 h shelf stations all start inside
    # the 2 h tolerance of the previous anchor -- the *first* genuine cast after the
    # on-station time belongs to this station, even when a later one is bigger.
    utc = np.datetime64("2022-03-04T02:16:45")
    spans = {
        "MA015": _span("2022-03-04T02:17:29", "2022-03-04T02:48:53", n=1813),
        "MA016": _span("2022-03-04T03:12:03", "2022-03-04T04:18:34", n=3839),  # bigger, later
    }
    rel, prov = A._match_master(utc, spans)
    assert rel == "MA015" and prov == "ctd-utc-nearest-cast-start"


def test_match_master_shallow_single_cast_still_resolves():
    # a genuinely small (shallow-shelf) cast that contains the utc must still match even
    # though it is below the stub threshold -- the size-blind safety net handles it.
    utc = np.datetime64("2025-09-18T22:11:00")
    spans = {"shallow": _span("2025-09-18T22:08", "2025-09-18T22:25", n=120)}
    rel, prov = A._match_master(utc, spans)
    assert rel == "shallow" and prov == "ctd-utc-in-master-window"


def test_match_master_nearest_within_tolerance_then_unmatched():
    # only a stub is forward in range -> size-blind fallback still picks it
    spans = {"m": _span("2025-10-03T06:40", "2025-10-03T07:30", n=50)}   # starts 14 min after
    utc = np.datetime64("2025-10-03T06:26:00")
    rel, prov = A._match_master(utc, spans)
    assert rel == "m" and prov == "ctd-utc-nearest-master-start"
    far = {"m": _span("2025-10-05T06:40", "2025-10-05T07:30")}           # > 2 h away
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


# --- flat-layout build (no MASTER/SLAVE subdirs) -------------------------------------
_FIX_LADCP = _REPO / "tests" / "fixtures" / "New_golden" / "Good" / "LADCP"


def test_build_index_flat_layout_classifies_by_facing(tmp_path):
    # one dir, opaque names with no master/slave signal -> the PD0 facing bit
    # must classify the heads and the pairing must still work
    import shutil
    flat = tmp_path / "LADCP"
    flat.mkdir()
    shutil.copy(_FIX_LADCP / "MORIA-80-LADCP-M.000", flat / "AAA0.000")
    shutil.copy(_FIX_LADCP / "MORIA-80-LADCP-S.000", flat / "BBB0.000")
    from ladcp.io.pd0 import read_pd0
    r = read_pd0(str(flat / "AAA0.000"), head="down", facing_hint="down")
    utc = str(r.time[min(10, r.n_ens - 1)].astype("datetime64[s]"))

    ctd = tmp_path / "CTD"
    ctd.mkdir()
    (ctd / "tx-01.hex").write_text(
        "* Sea-Bird SBE 9 Data File:\n"
        f"* NMEA UTC (Time) = "
        f"{np.datetime64(utc).item().strftime('%b %d %Y %H:%M:%S')}\n"
        "* NMEA Latitude = 62 09.68 N\n"
        "* NMEA Longitude = 011 31.85 W\n"
        "** Station: TX-01\n"
        "*END*\n")

    idx = A.build_index(flat, ctd, root=tmp_path, out=tmp_path / "idx.json")
    casts = idx["casts"]
    assert set(casts) == {"tx-01"}
    assert casts["tx-01"]["master"].endswith("AAA0.000")
    assert casts["tx-01"]["slave"].endswith("BBB0.000")
    facings = {pathlib.Path(rel).name: e["facing"]
               for rel, e in idx["scan_cache"].items()}
    assert facings == {"AAA0.000": "down", "BBB0.000": "up"}


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


_DATA = _REPO / "Data"   # full hex set incl. the fragmented MORIA 01-04
_HAS_FRAG = _DATA.exists() and (_RAW_LADCP / "MASTER").exists()


@pytest.mark.skipif(not _HAS_FRAG, reason="Data/ hex set + raw_ladcp_test/ not present")
def test_build_index_salvages_fragmented_casts(tmp_path):
    # stations 01-04 had the deck-unit logging restarted mid-cast: the right master/slave
    # pair must be picked by cast size + time, not by the stub nearest the on-station UTC.
    idx = A.build_index(_RAW_LADCP, _DATA, root=_REPO, out=tmp_path / "idx.json")
    casts = idx["casts"]
    expect = {
        "MORIA-01": ("M1910002.000", "S1910002.000", "ctd-utc-in-master-window"),
        "MORIA-02": ("M1910005.000", "S1910005.000", "ctd-utc-nearest-cast-start"),
        "MORIA-03": ("MLADC002.000", "SLADC003.000", "ctd-utc-nearest-cast-start"),
        "MORIA-04": ("MLADC005.000", "SLADC006.000", "ctd-utc-nearest-cast-start"),
    }
    for st, (m, s, prov) in expect.items():
        assert casts[st]["master"].endswith(m), f"{st} master -> {casts[st]['master']}"
        assert casts[st]["slave"].endswith(s), f"{st} slave -> {casts[st]['slave']}"
        assert casts[st]["provenance"] == prov
