"""Ship navigation tracks + SADCP clock-desync correction.

Shipboard-ADCP acquisition PCs are not always synchronised to GPS time: on
one validation cruise the VmDAS clock was ~12 years off (STA timestamps start 2010-01-01 for a
March-2022 cruise) while the embedded GPS *positions* were correct. Because the
ship's track is effectively a unique function of time, the wrong clock can be
recovered by sliding the SADCP position series against an independently
timestamped navigation track until they coincide:

    offset = argmin_dt  median | sadcp_pos(t_wrong + dt) - nav_pos(t_true) |

:func:`read_nav` loads the navigation track (SADO ``posicion`` exports or a
generic time/lat/lon CSV), :func:`estimate_time_offset` recovers the constant
clock offset, and :func:`shift_time` applies it to a
:class:`~ladcp.io.sadcp_vmdas.SadcpDataset`.
"""

from __future__ import annotations

from dataclasses import dataclass, replace
from pathlib import Path

import numpy as np

_M_PER_DEG_LAT = 110_540.0


@dataclass
class NavTrack:
    """An independently timestamped ship track (time-sorted)."""

    time: np.ndarray            # datetime64[ns]
    lat: np.ndarray
    lon: np.ndarray
    source: str = ""

    @property
    def n(self) -> int:
        return self.time.size


def _read_sado_posicion(path: Path):
    """One SADO ``*.posicion.txt`` / ``*_navsado.csv`` file -> (time, lat, lon).

    Header ``fecha,longitud,latitud,...``; rows ``DD-MM-YYYY HH:MM:SS,lon,lat,...``
    (the CSV variant uses ``;`` and ``DD/MM/YYYY``).
    """
    import pandas as pd

    head = path.read_text(encoding="latin-1", errors="replace")[:200]
    sep = ";" if head.count(";") > head.count(",") else ","
    df = pd.read_csv(path, sep=sep, encoding="latin-1")
    cols = {c.strip().lower(): c for c in df.columns}
    fecha, lon_c, lat_c = cols["fecha"], cols["longitud"], cols["latitud"]
    raw = df[fecha].astype(str).str.strip().str.replace("/", "-", regex=False)
    t = pd.to_datetime(raw, format="%d-%m-%Y %H:%M:%S", errors="coerce")
    lat = pd.to_numeric(df[lat_c], errors="coerce").to_numpy(float)
    lon = pd.to_numeric(df[lon_c], errors="coerce").to_numpy(float)
    ok = t.notna().to_numpy() & np.isfinite(lat) & np.isfinite(lon)
    return t.to_numpy("datetime64[ns]")[ok], lat[ok], lon[ok]


def _read_generic_csv(path: Path):
    """Generic nav CSV: any columns recognisable as time / lat / lon."""
    import pandas as pd

    df = pd.read_csv(path, encoding="latin-1")
    cols = {c.strip().lower(): c for c in df.columns}

    def pick(*names):
        for n in names:
            if n in cols:
                return cols[n]
        return None

    t_c = pick("time", "datetime", "date", "utc", "fecha")
    lat_c = pick("lat", "latitude", "latitud")
    lon_c = pick("lon", "longitude", "longitud")
    if not (t_c and lat_c and lon_c):
        raise ValueError(f"{path.name}: no time/lat/lon columns recognised "
                         f"(have {list(df.columns)})")
    t = pd.to_datetime(df[t_c], errors="coerce", dayfirst=False)
    lat = pd.to_numeric(df[lat_c], errors="coerce").to_numpy(float)
    lon = pd.to_numeric(df[lon_c], errors="coerce").to_numpy(float)
    ok = t.notna().to_numpy() & np.isfinite(lat) & np.isfinite(lon)
    return t.to_numpy("datetime64[ns]")[ok], lat[ok], lon[ok]


def read_nav(path: str | Path) -> NavTrack:
    """Load a navigation track from a file or a directory of files.

    Recognised formats (auto-detected per file): SADO ``posicion`` exports
    (``fecha,longitud,latitud,...``) and generic CSVs with time/lat/lon-like
    column names. A directory concatenates every parseable ``*.txt``/``*.csv``
    and sorts by time; files that don't parse are skipped (a directory often
    mixes formats -- e.g. SADO daily .txt next to .xls re-exports).
    """
    p = Path(path)
    files = sorted([*p.glob("*.txt"), *p.glob("*.csv")]) if p.is_dir() else [p]
    if not files:
        raise FileNotFoundError(f"no nav files under {p}")
    times, lats, lons = [], [], []
    for f in files:
        try:
            head = f.read_text(encoding="latin-1", errors="replace")[:200].lower()
            if "fecha" in head and "latitud" in head:
                t, la, lo = _read_sado_posicion(f)
            else:
                t, la, lo = _read_generic_csv(f)
        except (ValueError, KeyError):
            if not p.is_dir():
                raise
            continue
        times.append(t)
        lats.append(la)
        lons.append(lo)
    if not times:
        raise ValueError(f"no parseable nav files under {p}")
    t = np.concatenate(times)
    order = np.argsort(t)
    return NavTrack(time=t[order], lat=np.concatenate(lats)[order],
                    lon=np.concatenate(lons)[order], source=str(p))


def _track_misfit(dt_s: np.ndarray, t_wrong_s: np.ndarray, lat: np.ndarray,
                  lon: np.ndarray, nav_t_s: np.ndarray, nav_lat: np.ndarray,
                  nav_lon: np.ndarray, min_overlap: float) -> np.ndarray:
    """Median track distance [m] for each candidate offset (NaN if little overlap)."""
    m_per_deg_lon = _M_PER_DEG_LAT * np.cos(np.radians(np.nanmedian(nav_lat)))
    out = np.full(dt_s.size, np.nan)
    for k, dt in enumerate(dt_s):
        ts = t_wrong_s + dt
        inside = (ts >= nav_t_s[0]) & (ts <= nav_t_s[-1])
        if inside.mean() < min_overlap:
            continue
        nla = np.interp(ts[inside], nav_t_s, nav_lat)
        nlo = np.interp(ts[inside], nav_t_s, nav_lon)
        d = np.hypot((nla - lat[inside]) * _M_PER_DEG_LAT,
                     (nlo - lon[inside]) * m_per_deg_lon)
        out[k] = np.median(d)
    return out


def estimate_time_offset(time: np.ndarray, lat: np.ndarray, lon: np.ndarray,
                         nav: NavTrack, *, coarse_step: float = 60.0,
                         min_overlap: float = 0.5) -> dict:
    """Recover a wrong acquisition clock's constant offset from the nav track.

    ``time``/``lat``/``lon`` are the mis-clocked instrument's per-ensemble
    series (positions correct, times wrong). Scans every offset that maps the
    instrument's window onto the nav track with at least ``min_overlap`` of its
    ensembles covered, at ``coarse_step`` resolution, then refines to sub-second.
    Returns ``{"offset_s", "median_m", "p90_m", "overlap"}`` -- the residuals
    are the remaining track misfit at the chosen offset, i.e. how well the
    correction explains the data.
    """
    good = np.isfinite(lat) & np.isfinite(lon)
    if good.sum() < 10:
        raise ValueError("too few navigated ensembles to match against the track")
    t0 = time[good][0]
    t_s = (time[good] - t0) / np.timedelta64(1, "s")
    nav_t_s = (nav.time - t0) / np.timedelta64(1, "s")
    la, lo = lat[good], lon[good]

    # subsample the instrument series for the scan (the cost is smooth in dt)
    step = max(1, t_s.size // 2000)
    tc, lac, loc = t_s[::step], la[::step], lo[::step]

    lo_dt = nav_t_s[0] - t_s[-1]
    hi_dt = nav_t_s[-1] - t_s[0]
    coarse = np.arange(lo_dt, hi_dt + coarse_step, coarse_step)
    cost = _track_misfit(coarse, tc, lac, loc, nav_t_s, nav.lat, nav.lon,
                         min_overlap)
    if not np.isfinite(cost).any():
        raise ValueError(
            f"no candidate offset overlaps the nav track by >= {min_overlap:.0%}; "
            "wrong nav files for this instrument?")
    best = coarse[np.nanargmin(cost)]

    for span, stp in ((coarse_step, 5.0), (10.0, 0.5)):
        cand = np.arange(best - span, best + span + stp, stp)
        c = _track_misfit(cand, tc, lac, loc, nav_t_s, nav.lat, nav.lon,
                          min_overlap)
        best = cand[np.nanargmin(c)]

    # final residuals on the full series
    final = _track_misfit(np.array([best]), t_s, la, lo, nav_t_s,
                          nav.lat, nav.lon, 0.0)[0]
    ts = t_s + best
    inside = (ts >= nav_t_s[0]) & (ts <= nav_t_s[-1])
    m_per_deg_lon = _M_PER_DEG_LAT * np.cos(np.radians(np.nanmedian(nav.lat)))
    nla = np.interp(ts[inside], nav_t_s, nav.lat)
    nlo = np.interp(ts[inside], nav_t_s, nav.lon)
    d = np.hypot((nla - la[inside]) * _M_PER_DEG_LAT,
                 (nlo - lo[inside]) * m_per_deg_lon)
    return {
        "offset_s": float(best),
        "median_m": float(final),
        "p90_m": float(np.percentile(d, 90)) if d.size else float("nan"),
        "overlap": float(inside.mean()),
    }


def shift_time(ds, offset_s: float):
    """A copy of a :class:`~ladcp.io.sadcp_vmdas.SadcpDataset` with its clock shifted."""
    shifted = ds.time + np.timedelta64(round(offset_s * 1e9), "ns")
    return replace(ds, time=shifted,
                   source=f"{ds.source} [clock {offset_s:+.2f} s]")
