"""Optional raw-CTD ingest (#7): ``io/ctd_raw`` + the ``discover(from_hex=...)`` wiring.

The pure pieces (sibling-``.XMLCON`` resolution, graceful degradation when the raw
anchor or the optional CTD_project converter is missing) run everywhere. The
end-to-end ``.hex`` → ``.cnv`` → :class:`CTDTimeSeries` test runs only when
CTD_project (with its example cast) is available beside the repo.
"""

from __future__ import annotations

import numpy as np
import pytest

from ladcp import discovery as D
from ladcp.io import ctd_raw
from ladcp.io.ctd_cnv import read_ctd_cnv


def test_xmlcon_for_finds_sibling(tmp_path):
    hexp = tmp_path / "MORIA-01-CTD.hex"
    hexp.write_text("* hex\n")
    xml = tmp_path / "MORIA-01-CTD.XMLCON"
    xml.write_text("<config/>\n")
    assert ctd_raw.xmlcon_for(hexp) == xml


def test_xmlcon_for_matches_mismatched_case_stem(tmp_path):
    # the MORIA-25b..g case: .hex stem is lowercase, the .XMLCON carries an uppercase
    # station letter -> must still match on a case-sensitive filesystem.
    hexp = tmp_path / "MORIA-25b-CTD.hex"
    hexp.write_text("* hex\n")
    xml = tmp_path / "MORIA-25B-CTD.XMLCON"
    xml.write_text("<config/>\n")
    assert ctd_raw.xmlcon_for(hexp) == xml


def test_xmlcon_for_raises_when_absent(tmp_path):
    hexp = tmp_path / "only.hex"
    hexp.write_text("* hex\n")
    with pytest.raises(FileNotFoundError):
        ctd_raw.xmlcon_for(hexp)


def test_discover_from_hex_off_does_not_convert(tmp_path):
    """from_hex=False keeps the old behaviour: no .cnv found -> ctd is None."""
    index = {"casts": {"MORIA-99": {"master": str(tmp_path / "m.000"),
                                    "slave": None, "ctd_hex": str(tmp_path / "x.hex")}}}
    sf = D.discover("MORIA-99", root=tmp_path, index=index, from_hex=False)
    assert sf.ctd is None and sf.down.name == "m.000"


def test_discover_from_hex_without_anchor_is_graceful(tmp_path):
    """from_hex=True but the record has no ctd_hex -> ctd None, run still proceeds."""
    index = {"casts": {"MORIA-99": {"master": str(tmp_path / "m.000"),
                                    "slave": None}}}
    sf = D.discover("MORIA-99", root=tmp_path, index=index, from_hex=True)
    assert sf.ctd is None


def test_discover_prefers_existing_clean_cnv(tmp_path):
    """A pre-processed .cnv wins over conversion even when from_hex is set (Path B first)."""
    ctd_dir = tmp_path / "CTD"
    ctd_dir.mkdir()
    clean = ctd_dir / "moria-99_clean.cnv"
    clean.write_text("47.8 -7.8 10.0 0.0 12.0 35.5\n")
    index = {"casts": {"MORIA-99": {"master": str(tmp_path / "m.000"),
                                    "slave": None, "ctd_hex": str(tmp_path / "x.hex")}}}
    sf = D.discover("MORIA-99", root=tmp_path, index=index, ctd_dir=ctd_dir, from_hex=True)
    assert sf.ctd == clean


# --- end-to-end: needs CTD_project (optional) beside the repo ------------------------
def _example_hex():
    try:
        proj = ctd_raw._find_ctd_project()
    except RuntimeError:
        return None
    hexp = proj / "example_data" / "MORIA" / "raw" / "MORIA-01-CTD.hex"
    return hexp if hexp.exists() else None


@pytest.mark.skipif(_example_hex() is None,
                    reason="CTD_project example cast not available")
def test_cnv_from_hex_roundtrip(tmp_path):
    """Convert a real .hex and read it back through the standard CTD consumer."""
    out = ctd_raw.cnv_from_hex(_example_hex(), "MORIA-01", cache_dir=tmp_path)
    assert out.exists() and out.parent == tmp_path

    ts = read_ctd_cnv(str(out))
    assert ts.pressure.size > 100
    good = np.isfinite(ts.salinity)
    assert 30 < np.nanmedian(ts.salinity[good]) < 40
    # second call reuses the cache (no reconversion) and returns the same path
    again = ctd_raw.cnv_from_hex(_example_hex(), "MORIA-01", cache_dir=tmp_path)
    assert again == out
