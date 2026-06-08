"""Excel workbook writers (per-station + cruise) via pandas + openpyxl.

``openpyxl`` is the optional ``pyladcp[export]`` dependency; :func:`_require_openpyxl`
raises :class:`~ladcp.export.ExportDependencyError` with the install hint when it is missing,
so the CLI can warn and skip Excel while still emitting ODV/NetCDF/CSV (the same graceful
pattern as :func:`ladcp.io.ctd_raw._load_converter`).
"""

from __future__ import annotations

import pandas as pd

from . import ExportDependencyError
from .tables import (
    StationExport,
    bottomtrack_frame,
    locate_frame,
    metadata_dict,
    profile_frame,
    qa_frame,
    sadcp_frame,
    shear_frame,
    summary_row,
)


def _require_openpyxl() -> None:
    try:
        import openpyxl  # noqa: F401
    except ImportError as e:
        raise ExportDependencyError(
            "Excel export needs openpyxl: pip install pyladcp[export]") from e


def _meta_frame(export: StationExport) -> pd.DataFrame:
    md = metadata_dict(export)
    return pd.DataFrame({"key": list(md), "value": [md[k] for k in md]})


def write_station_xlsx(export: StationExport, path: str) -> str:
    """Write a station's profile/bottom-track/shear/sadcp/metadata/qa sheets. Returns ``path``."""
    _require_openpyxl()
    # data sheets carry station id + position so each is self-locating when extracted
    sheets = {
        "profile": locate_frame(profile_frame(export.result), export),
        "shear": locate_frame(shear_frame(export.result), export),
        "metadata": _meta_frame(export),
        "qa": qa_frame(export.qc),
    }
    bt = bottomtrack_frame(export.result)
    if bt is not None:
        sheets["bottom_track"] = locate_frame(bt, export)
    sv = sadcp_frame(export.result)
    if sv is not None:
        sheets["sadcp"] = locate_frame(sv, export)

    with pd.ExcelWriter(path, engine="openpyxl") as xw:
        for name in ("profile", "bottom_track", "shear", "sadcp", "metadata", "qa"):
            if name in sheets:
                sheets[name].to_excel(xw, sheet_name=name, index=False)
    return path


def write_cruise_xlsx(exports: list[StationExport], path: str, *, cruise: str = "") -> str:
    """Write a cruise workbook: stacked profiles + summary + metadata. Returns ``path``."""
    _require_openpyxl()
    profiles = []
    for exp in exports:
        profiles.append(locate_frame(profile_frame(exp.result), exp))
    profiles_df = (pd.concat(profiles, ignore_index=True) if profiles
                   else pd.DataFrame(columns=["station"]))
    summary_df = pd.DataFrame([summary_row(e) for e in exports])
    meta_df = pd.DataFrame([metadata_dict(e) for e in exports])

    with pd.ExcelWriter(path, engine="openpyxl") as xw:
        profiles_df.to_excel(xw, sheet_name="profiles", index=False)
        summary_df.to_excel(xw, sheet_name="summary", index=False)
        meta_df.to_excel(xw, sheet_name="metadata", index=False)
    return path
