"""Shared Simrad EK80 / SONAR-netCDF4 helpers (used by ek80_files + sadcp_ek80).

One home for the 1601 ("FILETIME") epoch conversions, the ADCP group probe, and the
file collector, so the two EK80 modules cannot drift apart. The three time converters
deliberately keep their distinct precision conventions (datetime-microsecond for the
timetable peeks, datetime64[ns] for the reader).
"""
from __future__ import annotations

import glob
import os
from datetime import datetime, timedelta, timezone

import numpy as np

EPOCH_1601 = datetime(1601, 1, 1, tzinfo=timezone.utc)   # SONAR-netCDF4 / FILETIME epoch
S_1601_TO_1970 = 11644473600                             # seconds 1601 -> unix epoch
ADCP_GROUP_CANDIDATES = [f"/Sonar/Beam_group{g}/ADCP" for g in range(1, 8)]


def ns1601_to_dt(v) -> datetime:
    """ns-since-1601 -> tz-aware ``datetime`` (microsecond precision)."""
    return EPOCH_1601 + timedelta(microseconds=int(v) / 1000.0)


def filetime_to_dt(low, high) -> datetime:
    """FILETIME (100-ns ticks since 1601, as two int32) -> tz-aware ``datetime``."""
    ticks = (high << 32) | (low & 0xFFFFFFFF)
    return EPOCH_1601 + timedelta(microseconds=ticks / 10.0)


def ek80_time_to_dt64(ns_since_1601) -> np.ndarray:
    """uint64 ns-since-1601 -> datetime64[ns] UTC (1601 is out of ns range, so offset)."""
    v = np.asarray(ns_since_1601).astype("uint64")
    ns70 = (v - np.uint64(S_1601_TO_1970 * 1_000_000_000)).astype("int64")
    return np.datetime64("1970-01-01T00:00:00", "ns") + ns70.astype("timedelta64[ns]")


def has_path(ds, path: str) -> bool:
    """Whether ``path`` resolves inside an open netCDF dataset."""
    try:
        ds[path]
        return True
    except (IndexError, KeyError):
        return False


def find_adcp_path(ds) -> str | None:
    """The beam group holding an ADCP ``Mean_current`` product, or ``None``."""
    return next((p for p in ADCP_GROUP_CANDIDATES if has_path(ds, p + "/Mean_current")),
                None)


def collect_files(paths, exts=("nc",)) -> list[str]:
    """Expand dirs (recursive ``*.<ext>``) and globs to a sorted, deduped file list."""
    files: list[str] = []
    for p in (paths if isinstance(paths, (list, tuple)) else [paths]):
        if os.path.isdir(p):
            for ext in exts:
                files += glob.glob(os.path.join(p, "**", f"*.{ext}"), recursive=True)
        else:
            files += glob.glob(p, recursive=True)
    return sorted(set(files))
