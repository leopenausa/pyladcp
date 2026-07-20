"""Per-station narrowing of a whole-cruise SADCP folder (hub EK80 layout).

``sadcp_profile`` reads ``<folder>/<station>/`` when that subdir exists (the layout
:func:`ladcp.hub.ek80_ops.extract_jobs` writes), so one ``[sadcp]`` folder in
``cruise.toml`` serves the whole cruise without each cast re-reading every other
cast's files. All quick-lane: only the folder-resolution helper is exercised.
"""

from __future__ import annotations

from ladcp.qa.pipeline import _narrow_to_station
from ladcp.session import SadcpConfig


def test_narrows_to_matching_subdir(tmp_path):
    (tmp_path / "MORIA2_05").mkdir()
    (tmp_path / "MORIA2_05" / "a.nc").write_bytes(b"")
    sa = SadcpConfig(folder=str(tmp_path), source="ek80")
    out = _narrow_to_station(sa, "MORIA2_05")
    assert out.folder == str(tmp_path / "MORIA2_05")
    assert out.source == "ek80"                   # identity otherwise untouched


def test_no_subdir_uses_folder_as_is(tmp_path):
    (tmp_path / "cruise.nc").write_bytes(b"")
    sa = SadcpConfig(folder=str(tmp_path), source="ek80")
    assert _narrow_to_station(sa, "MORIA2_05") is sa
    assert _narrow_to_station(sa, None) is sa


def test_empty_subdir_is_ignored(tmp_path):
    (tmp_path / "MORIA2_05").mkdir()              # exists but holds no .nc
    (tmp_path / "cruise.nc").write_bytes(b"")
    sa = SadcpConfig(folder=str(tmp_path), source="ek80")
    assert _narrow_to_station(sa, "MORIA2_05") is sa
