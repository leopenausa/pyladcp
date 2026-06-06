"""Parsers for legacy LDEO_IX golden outputs (`.lad`, `.bot`, `.log`).

Used as the reference for validating the Python re-implementation.
"""

from __future__ import annotations

import re
from dataclasses import dataclass

import numpy as np

_DEG_RE = re.compile(r"(\d+)\D+([\d.]+)'?\s*([NSEW])")


@dataclass
class GoldenProfile:
    header: dict
    z: np.ndarray
    u: np.ndarray
    v: np.ndarray
    err: np.ndarray


def _parse_pos(s: str) -> float:
    """'46°N 29.1216'' -> decimal degrees (signed)."""
    m = _DEG_RE.search(s)
    if not m:
        return float("nan")
    deg, minutes, hemi = float(m.group(1)), float(m.group(2)), m.group(3)
    val = deg + minutes / 60.0
    return -val if hemi in ("S", "W") else val


def _read_columnar(path: str) -> GoldenProfile:
    header: dict = {}
    rows = []
    with open(path) as fh:
        for line in fh:
            if "=" in line and not line.lstrip()[:1].isdigit():
                key, _, val = line.partition("=")
                header[key.strip()] = val.strip()
            else:
                parts = line.split()
                if len(parts) >= 4:
                    try:
                        rows.append([float(p) for p in parts[:4]])
                    except ValueError:
                        continue
    arr = np.array(rows) if rows else np.empty((0, 4))
    # signed positions / declination convenience fields
    if "Start_Lat" in header:
        header["lat"] = _parse_pos(header["Start_Lat"])
    if "Start_Lon" in header:
        header["lon"] = _parse_pos(header["Start_Lon"])
    if "Deviation" in header:
        header["deviation"] = float(header["Deviation"])
    if "Bottom depth" in header:
        header["bottom_depth"] = float(header["Bottom depth"])
    return GoldenProfile(header, arr[:, 0], arr[:, 1], arr[:, 2], arr[:, 3])


def read_lad(path: str) -> GoldenProfile:
    """Final absolute-velocity profile: columns z u v ev."""
    return _read_columnar(path)


def read_bot(path: str) -> GoldenProfile:
    """Bottom-track referenced profile: columns z u v err."""
    return _read_columnar(path)


def scan_log(path: str) -> dict:
    """Extract a few scalar checkpoints from the legacy `.log`."""
    out: dict = {}
    patterns = {
        "declination": r"magnetic declination of\s*([-\d.]+)",
        "dz": r"vertical resolution \(ps\.dz\) is\s*([-\d.]+)",
        "barotropic_vel_err": r"Barotropic velocity error\s*([-\d.]+)",
        "superensemble_vel_err": r"super ensemble velocity error\s*([-\d.]+)",
        "ctd_max_depth": r"CTD max depth\s*:\s*([-\d.]+)",
        "rdi_bottom_velocities": r"found\s*(\d+)\s*finite RDI bottom velocities",
    }
    text = open(path, errors="ignore").read()
    for key, pat in patterns.items():
        m = re.search(pat, text)
        if m:
            out[key] = float(m.group(1))
    out["sadcp_used"] = ("no SADCP data used" not in text and "no SADCP data" not in text)
    return out
