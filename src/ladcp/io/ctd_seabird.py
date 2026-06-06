"""Sea-Bird 24 Hz CTD/nav loader for the GO-SHIP golden (RB1606/P18).

The GO-SHIP input is a full Sea-Bird ``.cnv`` at ~24 Hz with **per-scan lat/lon
appended** (cols ``scan, timeJ, timeS, prDM, t090C, …, sal00, …, latitude, longitude``).
The MORIA pre-cleaned 6-column 1 Hz file (:mod:`ladcp.io.ctd`) does not apply here.

This module re-derives the CTD/nav side of the LDEO_IX driver from the MATLAB
``loadctd.m``/``loadnav.m``, with one piece that lives *outside* LDEO_IX: the decimation
of the raw 24 Hz cast to the ~1 Hz time series the golden actually consumed (Thurnherr's
``.1Hz`` preprocessing). We reproduce that by bin-averaging on integer elapsed seconds;
it is validated by matching the golden ``ctd_t``/``ctd_s`` on the output ``z`` grid.

What is faithful to ``loadctd.m``:
- ``p2z`` Saunders & Fofonoff pressure->depth (z=0 at p=0), identical coefficients.
- iterative 3-sigma vertical-velocity pressure-spike rejection.
- downcast CTD profile = sort by pressure to ``pmax`` then interpolate.

What is **deferred to Phase 3** (needs the raw ADCP ``w``/``time``):
- ``besttlag`` time-lag matching of ADCP to CTD and the ADCP-time shift.
- interpolation of CTD onto ADCP ping times.
So this module stops at the per-scan CTD/nav series + the downcast profile; the lag join
is wired in once :mod:`ladcp.io.pd0` ingestion lands.
"""

from __future__ import annotations

import re
from dataclasses import dataclass, field
from datetime import datetime
from typing import Any

import numpy as np

from ..models import CTDTimeSeries

# Sea-Bird marks bad data with the sentinel value -9.990e-29 (a specific magnitude
# ~1e-29, NOT a threshold — real data such as longitude -110 are ordinary negatives).
# Some files also use the legacy -9e99 huge-negative flag. ``flag`` column != 0 is bad too.
_SBE_BADFLAG = -9.99e-29


def _mask_bad(v: np.ndarray) -> np.ndarray:
    """NaN out Sea-Bird bad sentinels in-place-safe; returns a float copy."""
    v = v.astype(float)
    bad = np.isclose(v, _SBE_BADFLAG, rtol=1e-2, atol=0.0) | (v <= -1e30)
    v[bad] = np.nan
    return v


@dataclass
class SeabirdCast:
    """One Sea-Bird cast: raw header + decimated 1 Hz CTD/nav series.

    ``ctd`` is the despiked 1 Hz :class:`CTDTimeSeries` (elapsed seconds base). ``header``
    carries the parsed column map, scan count, and cast start datetime (UTC).
    """

    header: dict[str, Any]
    ctd: CTDTimeSeries
    n_raw_scans: int
    n_spikes_removed: int = 0
    meta: dict[str, Any] = field(default_factory=dict)


# --------------------------------------------------------------------------- #
# Header parsing
# --------------------------------------------------------------------------- #
_NAME_RE = re.compile(r"^#\s*name\s+(\d+)\s*=\s*([^:]+):", re.IGNORECASE)
_NVAL_RE = re.compile(r"^#\s*nvalues\s*=\s*(\d+)", re.IGNORECASE)
_UPLOAD_RE = re.compile(r"System UpLoad Time\s*=\s*(.+)$", re.IGNORECASE)
_NMEA_TIME_RE = re.compile(r"NMEA UTC \(Time\)\s*=\s*(.+)$", re.IGNORECASE)


def parse_cnv_header(path: str) -> dict[str, Any]:
    """Parse a Sea-Bird ``.cnv`` header into column map + metadata.

    Returns a dict with ``columns`` ({short_name: 0-based index}), ``nvalues``,
    ``start_time`` (datetime, UTC; prefers NMEA UTC over upload time), ``n_header``
    (number of leading ``*``/``#`` lines), and the raw header lines.
    """
    columns: dict[str, int] = {}
    nvalues = None
    upload_time = nmea_time = None
    n_header = 0
    raw: list[str] = []
    with open(path, errors="ignore") as fh:
        for ln in fh:
            if ln[:1] not in ("*", "#"):
                break
            n_header += 1
            raw.append(ln.rstrip("\n"))
            m = _NAME_RE.match(ln)
            if m:
                columns[m.group(2).strip()] = int(m.group(1))
                continue
            m = _NVAL_RE.match(ln)
            if m:
                nvalues = int(m.group(1))
                continue
            m = _UPLOAD_RE.search(ln)
            if m:
                upload_time = _parse_sbe_time(m.group(1))
                continue
            m = _NMEA_TIME_RE.search(ln)
            if m:
                nmea_time = _parse_sbe_time(m.group(1))
    return {
        "columns": columns,
        "nvalues": nvalues,
        "start_time": nmea_time or upload_time,
        "upload_time": upload_time,
        "nmea_time": nmea_time,
        "n_header": n_header,
        "raw": raw,
    }


def _parse_sbe_time(s: str) -> datetime | None:
    s = s.strip()
    for fmt in ("%b %d %Y %H:%M:%S", "%b %d %Y  %H:%M:%S"):
        try:
            return datetime.strptime(re.sub(r"\s+", " ", s), "%b %d %Y %H:%M:%S")
        except ValueError:
            continue
    return None


def _find_col(columns: dict[str, int], *candidates: str) -> int:
    for c in candidates:
        if c in columns:
            return columns[c]
    raise KeyError(f"none of {candidates} found in cnv columns {list(columns)}")


# --------------------------------------------------------------------------- #
# Physics: pressure -> depth (legacy loadctd.m / p2z)
# --------------------------------------------------------------------------- #
def p2z(p: np.ndarray, lat: float) -> np.ndarray:
    """Pressure [dbar] -> depth [m, positive down] (Saunders & Fofonoff 1976).

    Faithful port of the ``p2z`` subfunction in ``loadctd.m``: **z=0 at p=0** (not 1 atm
    at the surface). ``lat`` in degrees.
    """
    p = np.asarray(p, dtype=float) / 10.0  # legacy divides dbar by 10
    x = np.sin(lat / 57.29578) ** 2
    gr = 9.780318 * (1.0 + (5.2788e-3 + 2.36e-5 * x) * x) + 1.092e-5 * p
    depth = (((-1.82e-11 * p + 2.279e-7) * p - 2.2512e-3) * p + 97.2659) * p
    return depth / gr


# --------------------------------------------------------------------------- #
# Loader
# --------------------------------------------------------------------------- #
def load_seabird_cnv(
    path: str,
    *,
    lat_ref: float | None = None,
    despike: bool = True,
) -> SeabirdCast:
    """Load + decimate a 24 Hz Sea-Bird ``.cnv`` to a 1 Hz CTD/nav series.

    Steps (CTD/nav portion of LDEO_IX, ADCP-independent):
      1. parse header, locate ``timeS``/``prDM``/``t090C``/``sal00``/lat/lon columns;
      2. read scans, NaN out Sea-Bird bad sentinels;
      3. bin-average to 1 Hz on integer elapsed seconds (the ``.1Hz`` decimation);
      4. iterative 3-sigma vertical-velocity pressure-spike rejection (loadctd.m).

    ``lat_ref`` (deg) is used for the ``p2z`` despike; defaults to the median nav latitude.
    """
    hdr = parse_cnv_header(path)
    cols = hdr["columns"]
    i_time = _find_col(cols, "timeS")
    i_pr = _find_col(cols, "prDM", "prdM", "pr")
    i_t = _find_col(cols, "t090C", "tv290C", "t068C")
    i_s = _find_col(cols, "sal00", "sal11")
    i_lat = _find_col(cols, "latitude")
    i_lon = _find_col(cols, "longitude")
    i_flag = cols.get("flag")

    # --- read data block ------------------------------------------------- #
    raw = np.loadtxt(path, comments=("*", "#"))
    if raw.ndim == 1:
        raw = raw[None, :]
    n_raw = raw.shape[0]

    def col(i: int) -> np.ndarray:
        return _mask_bad(raw[:, i])

    timeS = raw[:, i_time].astype(float)
    pr, t, s = col(i_pr), col(i_t), col(i_s)
    lat, lon = col(i_lat), col(i_lon)
    if i_flag is not None:
        bad = raw[:, i_flag].astype(float) != 0.0
        for v in (pr, t, s):
            v[bad] = np.nan

    # --- decimate to 1 Hz on integer elapsed seconds --------------------- #
    sec = np.floor(timeS - timeS[0]).astype(int)
    nbin = int(sec.max()) + 1

    def binavg(v: np.ndarray) -> np.ndarray:
        tot = np.zeros(nbin)
        cnt = np.zeros(nbin)
        np.add.at(tot, sec, np.nan_to_num(v))
        np.add.at(cnt, sec, np.isfinite(v))
        with np.errstate(invalid="ignore", divide="ignore"):
            out = tot / cnt
        out[cnt == 0] = np.nan
        return out

    te = np.arange(nbin, dtype=float)  # 1 Hz elapsed seconds
    P, T, S = binavg(pr), binavg(t), binavg(s)
    LA, LO = binavg(lat), binavg(lon)

    if lat_ref is None:
        lat_ref = float(np.nanmedian(LA))

    # --- iterative 3-sigma pressure-spike rejection (loadctd.m) ---------- #
    n_spikes = 0
    if despike:
        keep = np.ones(nbin, dtype=bool)
        for _ in range(10):
            idx = np.where(keep)[0]
            if idx.size < 3:
                break
            z = -p2z(P[idx], lat_ref)
            dt = np.diff(te[idx])
            wctd = -np.diff(z) / dt
            sd = np.nanstd(wctd)
            bad_local = np.abs(wctd) > 3 * sd
            bad = np.zeros(idx.size, dtype=bool)
            bad[:-1] |= bad_local
            bad[1:] |= bad_local  # legacy removes the spike and its neighbour
            nb = int(np.count_nonzero(bad))
            if nb <= 9:
                break
            keep[idx[bad]] = False
            n_spikes += nb
        sel = keep
        te, P, T, S, LA, LO = te[sel], P[sel], T[sel], S[sel], LA[sel], LO[sel]

    ctd = CTDTimeSeries(
        time_elapsed_s=te,
        lat=LA,
        lon=LO,
        pressure=P,
        temperature=T,
        salinity=S,
        meta={"lat_ref": lat_ref, "source": path},
    )
    return SeabirdCast(
        header=hdr,
        ctd=ctd,
        n_raw_scans=n_raw,
        n_spikes_removed=n_spikes,
        meta={"decimation": "1Hz bin-average on integer elapsed seconds"},
    )


# --------------------------------------------------------------------------- #
# Downcast profile (loadctd.m: sort by pressure to pmax, interpolate)
# --------------------------------------------------------------------------- #
def downcast_profile(
    cast: SeabirdCast, lat: float | None = None
) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    """Build the downcast CTD profile ``(z, p, t, s)`` sorted by depth.

    Mirrors ``loadctd.m``: take scans up to maximum pressure (the downcast), convert
    pressure to depth via :func:`p2z`, and sort by depth. ``z`` positive-down.
    """
    c = cast.ctd
    lat = c.meta.get("lat_ref") if lat is None else lat
    P, T, S = c.pressure, c.temperature, c.salinity
    imax = int(np.nanargmax(P))
    p = P[: imax + 1]
    t = T[: imax + 1]
    s = S[: imax + 1]
    z = p2z(p, lat)
    order = np.argsort(z)
    return z[order], p[order], t[order], s[order]


def profile_on_z(
    cast: SeabirdCast, z_grid: np.ndarray, lat: float | None = None
) -> tuple[np.ndarray, np.ndarray]:
    """Interpolate downcast ``(t, s)`` onto an output depth grid (positive-down m).

    This is the comparison hook against the golden ``ctd_t``/``ctd_s`` arrays.
    """
    z, _p, t, s = downcast_profile(cast, lat)
    gt = np.isfinite(z) & np.isfinite(t)
    gs = np.isfinite(z) & np.isfinite(s)
    ti = np.interp(z_grid, z[gt], t[gt], left=np.nan, right=np.nan)
    si = np.interp(z_grid, z[gs], s[gs], left=np.nan, right=np.nan)
    return ti, si


def nav_position(cast: SeabirdCast, when: str = "start") -> tuple[float, float]:
    """Median lat/lon over the first/last minute of the cast (≈ loadnav nav_start/end)."""
    c = cast.ctd
    n = c.lat.size
    win = slice(0, min(120, n)) if when == "start" else slice(max(0, n - 120), n)
    return float(np.nanmedian(c.lat[win])), float(np.nanmedian(c.lon[win]))
