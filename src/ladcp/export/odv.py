"""Ocean Data View Generic Spreadsheet writer (cruise-stacked).

Emits a tab-separated file ODV imports directly: ``//`` provenance comment lines, a
metadata + data column header, then one row per (station, depth bin) with every station
stacked. Following ODV's convention the metadata columns (Cruise … Bot. Depth) are written
only on a station's first sample row and left blank on its continuation rows, so ODV groups
the rows into casts. Time is the unambiguous ISO 8601 column ODV recognises natively.
"""

from __future__ import annotations

from pathlib import Path

import numpy as np

from .tables import StationExport, _pyladcp_version, metadata_dict, profile_frame

# metadata columns (per cast) then data columns (per sample)
_META_COLS = [
    "Cruise", "Station", "Type", "yyyy-mm-ddThh:mm:ss.sss",
    "Longitude [degrees_east]", "Latitude [degrees_north]", "Bot. Depth [m]",
]
_DATA_COLS = [
    "Depth [m]", "Eastward velocity [m/s]", "Northward velocity [m/s]",
    "Velocity error [m/s]",
]


def _fmt(x: float, nd: int = 4) -> str:
    return "" if x is None or not np.isfinite(x) else f"{x:.{nd}f}"


def write_odv(exports: list[StationExport], path: str, *, cruise: str = "") -> str:
    """Write the stations in ``exports`` as one ODV Generic Spreadsheet. Returns ``path``."""
    lines = [
        "//<Encoding>UTF-8</Encoding>",
        f"//<Creator>pyladcp {_pyladcp_version()}</Creator>",
        f"//<Cruise>{cruise}</Cruise>",
        "//LADCP absolute ocean velocity profiles (eastward/northward, m/s).",
        "\t".join(_META_COLS + _DATA_COLS),
    ]
    for exp in exports:
        md = metadata_dict(exp)
        df = profile_frame(exp.result)
        first = True
        for _, row in df.iterrows():
            if first:
                meta = [
                    md["cruise"], md["station"], "C",
                    md["time_utc"],
                    _fmt(md["longitude_deg"], 5), _fmt(md["latitude_deg"], 5),
                    _fmt(md["bottom_depth_m"], 1),
                ]
                first = False
            else:
                meta = [""] * len(_META_COLS)
            data = [
                _fmt(row["depth_m"], 1), _fmt(row["u_ms"]), _fmt(row["v_ms"]),
                _fmt(row["uerr_ms"]),
            ]
            lines.append("\t".join(meta + data))

    text = "\n".join(lines) + "\n"
    Path(path).write_text(text, encoding="utf-8")
    return path
