"""Data-driven ingest/config (#4a): param resolution, file discovery, CTD header map.

These exercise the framework that lets the pipeline load datasets beyond the hard-wired
MORIA-05 preset: the cruise param resolver, the time-based slave pairing, the bare-id ->
label normalisation, and the Seabird ``.cnv`` column auto-mapping. The pure pieces run
everywhere; the header-config check uses the committed MORIA-80 fixture.
"""

from __future__ import annotations

import pathlib

import numpy as np
import pytest

from ladcp import discovery as D
from ladcp.config import CRUISES, resolve_params
from ladcp.io.ctd_cnv import _parse_cnv_header, read_ctd_cnv

ROOT = pathlib.Path(__file__).resolve().parent / "fixtures"
GOOD = ROOT / "New_golden" / "Good"
DOWN = GOOD / "LADCP" / "MORIA-80-LADCP-M.000"
UP = GOOD / "LADCP" / "MORIA-80-LADCP-S.000"
CTD = GOOD / "CTD" / "moria-80_clean.cnv"
_HAS_FIX = DOWN.exists()


# --- resolve_params (Step 2) --------------------------------------------------------
def test_resolve_params_stamps_station_from_cruise_preset():
    p = resolve_params("MORIA", "MORIA-10")
    assert p.station == "MORIA-10"
    assert p.cruise_id == "MORIA"
    assert p.pglim == 50.0 and p.dz == 8.0          # MORIA operator defaults


def test_resolve_params_is_case_insensitive():
    assert resolve_params("moria", "MORIA-10").pglim == 50.0


def test_resolve_params_applies_overrides_last():
    p = resolve_params("MORIA", "MORIA-10", overrides={"drot": -3.5, "pglim": 25.0})
    assert p.drot == -3.5 and p.pglim == 25.0


def test_resolve_params_unknown_cruise_falls_back_to_generic():
    # a cruise without a preset must still process (and label exports) correctly:
    # generic operator defaults, its own cruise_id, no MORIA-specific layers
    p = resolve_params("CRUISE2", "t1-05")
    assert p.cruise_id == "CRUISE2" and p.station == "t1-05"
    assert p.pglim == 50.0 and p.cut == 7.0 and p.btrk_mode == 3
    assert p.edit_nearfield_dn_bins == ()        # the monocorer mask is MORIA's, not generic
    assert p.sadcp == 0 and p.drot is None


def test_resolve_params_rejects_unknown_field():
    with pytest.raises(AttributeError):
        resolve_params("MORIA", "X", overrides={"not_a_field": 1})


def test_moria_cruise_registered():
    assert "MORIA" in CRUISES


# --- discovery: time-based slave pairing (Step 3) -----------------------------------
def _span(a: str, b: str):
    return np.datetime64(a), np.datetime64(b)


def test_best_overlap_picks_maximum_overlap():
    master = _span("2025-09-19T10:35", "2025-09-19T11:18")
    cands = {
        "early": _span("2025-09-19T07:21", "2025-09-19T07:54"),   # SLADC012-like, no overlap
        "match": _span("2025-09-19T10:35", "2025-09-19T11:18"),   # SLADC013-like, full overlap
        "later": _span("2025-09-19T14:06", "2025-09-19T15:18"),
    }
    assert D.best_overlap(master, cands) == "match"


def test_best_overlap_none_when_disjoint():
    master = _span("2025-09-19T10:00", "2025-09-19T11:00")
    cands = {"x": _span("2025-09-20T10:00", "2025-09-20T11:00")}
    assert D.best_overlap(master, cands) is None


def test_normalize_bare_id_to_label():
    assert D._normalize("MORIA", "10") == "MORIA-10"
    assert D._normalize("MORIA", "MORIA-10") == "MORIA-10"      # label passes through
    assert D._normalize("MORIA", "CRUISE2_002") == "CRUISE2_002"


def test_moria10_in_manifest_with_time_paired_slave():
    entry = D.MANIFESTS["MORIA"]["MORIA-10"]
    assert entry.master.endswith("MLADC012.000")        # from the cast log
    assert entry.slave_dir.endswith("SLAVE")            # slave resolved by time, not index


# --- CTD .cnv header auto-map (Step 4) ----------------------------------------------
def test_parse_cnv_header_maps_sbe_roles(tmp_path):
    cnv = tmp_path / "h.cnv"
    cnv.write_text(
        "* Sea-Bird SBE 9 Data File\n"
        "# name 0 = latitude: Latitude [deg]\n"
        "# name 1 = longitude: Longitude [deg]\n"
        "# name 2 = prDM: Pressure, Digiquartz [db]\n"
        "# name 3 = timeS: Time, Elapsed [seconds]\n"
        "# name 4 = t090C: Temperature [ITS-90, deg C]\n"
        "# name 5 = sal00: Salinity, Practical [PSU]\n"
        "# end\n"
        "47.8 -7.8 10.0 0.0 12.0 35.5\n"
        "47.8 -7.8 12.0 1.0 12.1 35.6\n"
    )
    n_header, roles = _parse_cnv_header(str(cnv))
    assert n_header == 8                                 # 1 '*' + 6 'name' + 1 'end'
    assert roles == {"lat": 0, "lon": 1, "pressure": 2,
                     "time": 3, "temperature": 4, "salinity": 5}
    ts = read_ctd_cnv(str(cnv))
    assert ts.pressure[1] == 12.0 and ts.salinity[1] == 35.6
    assert ts.meta["header_lines"] == 8


def test_headerless_cnv_uses_param_field_map():
    # the MORIA clean files carry no header -> fall back to CastParams defaults
    if not _HAS_FIX:
        pytest.skip("MORIA-80 fixture not present")
    n_header, roles = _parse_cnv_header(str(CTD))
    assert n_header == 0 and roles == {}
    ts = read_ctd_cnv(str(CTD))
    assert ts.pressure.size > 100 and np.isfinite(ts.lat).any()


# --- header-derived instrument config (Step 1) --------------------------------------
@pytest.mark.skipif(not _HAS_FIX, reason="MORIA-80 fixture not present")
def test_apply_header_config_sets_head_count_and_stashes_geometry():
    from ladcp.qa.ingest import apply_header_config, load_dualhead

    p = resolve_params("MORIA", "MORIA-80")
    dh = load_dualhead(str(DOWN), str(UP), station="MORIA-80", params=p)
    apply_header_config(p, dh)
    assert p.up_dn_looker == 1                          # both heads present, down faces down
    assert p.extra["instrument"]["nbin_d"] == dh.down.n_cells


# --- discovery failures are real exceptions (not SystemExit) ------------------------
def test_discover_unknown_station_raises_filenotfound(tmp_path):
    """Servers (ladcp-studio) must be able to catch discovery failures; SystemExit
    is a BaseException and escaped the HTTP error mapping as a raw 500."""
    (tmp_path / "LADCP").mkdir()
    with pytest.raises(FileNotFoundError, match=r"no down-looker found for station '-'"):
        D.discover("-", root=tmp_path, cruise="NOPE")


def test_discover_ambiguous_match_raises_valueerror(tmp_path):
    lad = tmp_path / "LADCP"
    lad.mkdir()
    (lad / "X-80a-LADCP-M.000").write_bytes(b"")
    (lad / "X-80b-LADCP-M.000").write_bytes(b"")
    with pytest.raises(ValueError, match="ambiguous match"):
        D.discover("80", root=tmp_path, cruise="X")
