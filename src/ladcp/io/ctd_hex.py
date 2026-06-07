"""Seabird ``.hex`` / ``.hdr`` header reader -- the discovery anchor (#4a).

A raw LADCP PD0 file carries time but no station number, so the archive index needs an
external anchor to label each cast. The Seabird CTD raw files supply it: the ASCII header
block (identical at the top of the ``.hex`` and in the sidecar ``.hdr``) records the
station, the absolute NMEA UTC of the cast, and the GPS position -- everything needed to
tie a station to a raw LADCP file by time. Only the header is read; the binary scan data is
never decoded.

Example header lines::

    * NMEA UTC (Time) = Oct 03 2025  06:25:56
    * NMEA Latitude = 62 09.68 N
    * NMEA Longitude = 011 31.85 W
    ** Station:   ST80
    ** Depth:   1086
"""

from __future__ import annotations

import re
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path

import numpy as np

# absolute-time fields in preference order: GPS UTC first, then the deck-unit clocks
_TIME_KEYS = ("NMEA UTC (Time)", "System UTC", "System UpLoad Time")
_TIME_FMT = "%b %d %Y %H:%M:%S"           # "Oct 03 2025 06:25:56" (whitespace collapsed)


@dataclass(frozen=True)
class CtdCastHeader:
    """The anchor fields parsed from one Seabird CTD header."""

    station: str                # normalised label, e.g. "MORIA-80"
    raw_station: str            # operator field verbatim, e.g. "ST80"
    utc: np.datetime64          # absolute cast time (NMEA UTC preferred)
    lat: float                  # decimal degrees (+N)
    lon: float                  # decimal degrees (+E)
    depth: float | None         # logged sounding [m]
    cruise: str
    time_source: str            # which header field supplied utc
    path: Path


def _dm_to_deg(s: str) -> float:
    """`62 09.68 N` -> +62.1613 decimal degrees."""
    m = re.match(r"\s*(\d+)\s+([\d.]+)\s*([NSEW])", s)
    if not m:
        raise ValueError(f"unparseable NMEA position {s!r}")
    deg = float(m.group(1)) + float(m.group(2)) / 60.0
    return -deg if m.group(3).upper() in ("S", "W") else deg


def _parse_time(value: str) -> np.datetime64:
    """`Oct 03 2025  06:25:56` -> datetime64[s] (whitespace-tolerant)."""
    dt = datetime.strptime(" ".join(value.split()), _TIME_FMT).replace(tzinfo=timezone.utc)
    return np.datetime64(int(dt.timestamp()), "s")


def _station_from_name(path: Path) -> str:
    """`MORIA-80-CTD.hex` -> `MORIA-80` (drop a trailing -CTD/_CTD token)."""
    stem = path.stem
    return re.sub(r"[-_ ]*ctd$", "", stem, flags=re.IGNORECASE) or stem


def read_hex_header(path: str | Path) -> CtdCastHeader:
    """Parse the Seabird header of a ``.hex`` (or ``.hdr``) into a :class:`CtdCastHeader`.

    Reads only the leading ``*`` / ``**`` comment block (up to ``*END*`` or the first data
    line). The station label comes from the filename, with the ``** Station`` field kept as
    ``raw_station``; the timestamp uses the first available of NMEA UTC, System UTC, System
    UpLoad Time.
    """
    path = Path(path)
    star: dict[str, str] = {}       # "* key = value"
    dstar: dict[str, str] = {}      # "** key: value"
    with open(path, encoding="latin-1") as fh:
        for line in fh:
            s = line.rstrip("\n")
            if s.startswith("*END*"):
                break
            if not s.startswith("*"):
                break               # reached binary/data before *END* (defensive)
            if s.startswith("**"):
                key, _, val = s[2:].partition(":")
                if val:
                    dstar[key.strip()] = val.strip()
            else:
                key, _, val = s[1:].partition("=")
                if val:
                    star[key.strip()] = val.strip()

    time_source = next((k for k in _TIME_KEYS if k in star), None)
    if time_source is None:
        raise ValueError(f"{path}: no absolute-time field {_TIME_KEYS} in header")
    if "NMEA Latitude" not in star or "NMEA Longitude" not in star:
        raise ValueError(f"{path}: no NMEA position in header")

    depth = None
    if "Depth" in dstar:
        try:
            depth = float(re.findall(r"[\d.]+", dstar["Depth"])[0])
        except (IndexError, ValueError):
            depth = None

    return CtdCastHeader(
        station=_station_from_name(path),
        raw_station=dstar.get("Station", ""),
        utc=_parse_time(star[time_source]),
        lat=_dm_to_deg(star["NMEA Latitude"]),
        lon=_dm_to_deg(star["NMEA Longitude"]),
        depth=depth,
        cruise=dstar.get("Cruise", ""),
        time_source=time_source,
        path=path,
    )


def scan_ctd_dir(folder: str | Path, *, pattern: str = "*.hex") -> list[CtdCastHeader]:
    """Read every CTD header under ``folder`` (``.hex`` by default; ``.hdr`` also works)."""
    folder = Path(folder)
    out: list[CtdCastHeader] = []
    for p in sorted(folder.glob(pattern)):
        try:
            out.append(read_hex_header(p))
        except ValueError:
            continue
    return out
