"""File-level utilities for Simrad EK80 ADCP-mode datasets.

Two jobs, both designed for **big files on a slow/remote share** (an EK80 current run
is hundreds of ~0.5 GB SONAR-netCDF4 files):

* :func:`scan` — build a low-IO ``[start, end]`` time table. Start comes from the
  filename (``...-D<YYYYMMDD>-T<HHMMSS>``, zero IO); end is read by peeking *only* the
  time coordinate of a ``.nc`` (one HDF5 chunk) or the first+last datagram header of a
  ``.raw`` (a few hundred bytes). :func:`correlate` then maps each LADCP cast to the
  EK80 file(s) overlapping its on-station window.
* :func:`slim_extract` — copy *only* the ADCP current group
  (``/Sonar/Beam_group*/ADCP``) out of a full file into a ~30–100 MB local file,
  vlen-faithful, so :mod:`ladcp.io.sadcp_ek80` reads the slim copy and the original
  identically.

The ``ladcp-ek80`` command (:mod:`ladcp.ek80_cli`) wraps both.
"""
from __future__ import annotations

import bisect
import glob
import os
import re
import struct
from datetime import datetime, timedelta, timezone

FNAME_RE = re.compile(r"-D(\d{8})-T(\d{6})")
EPOCH_1601 = datetime(1601, 1, 1, tzinfo=timezone.utc)   # SONAR-netCDF4 / FILETIME epoch
_S_1601_TO_1970 = 11644473600
_ADCP_CANDIDATES = [f"/Sonar/Beam_group{g}/ADCP" for g in range(1, 8)]

# variables kept by the slim extract (everything sadcp_ek80.read_ek80 needs)
_KEEP_MEANCURRENT = ["mean_time", "current_velocity_geographical_east",
                     "current_velocity_geographical_north", "current_velocity_geographical_down",
                     "quality", "averaging", "mean_platform_latitude",
                     "mean_platform_longitude", "mean_bin_length", "percent_good_limit"]
_KEEP_ADCP = ["vertical_sample_interval", "depth_first_sample_center", "error_velocity",
              "bin_length", "correlation_factor_limit", "error_velocity_limit"]


# --------------------------------------------------------------------------- scan
def start_from_name(path: str):
    """Start datetime parsed from the EK80 filename, or ``None`` if it doesn't match."""
    m = FNAME_RE.search(os.path.basename(path))
    if not m:
        return None
    return datetime.strptime(m.group(1) + m.group(2), "%Y%m%d%H%M%S").replace(tzinfo=timezone.utc)


def _ns1601_to_dt(v):
    return EPOCH_1601 + timedelta(microseconds=int(v) / 1000.0)


def _filetime_to_dt(low, high):
    ticks = (high << 32) | (low & 0xFFFFFFFF)
    return EPOCH_1601 + timedelta(microseconds=ticks / 10.0)


def _has_path(ds, path: str) -> bool:
    try:
        ds[path]
        return True
    except (IndexError, KeyError):
        return False


def peek_nc_time(path: str):
    """``(start, end, n, lat, lon)`` from a ``.nc``, reading only the ADCP time coord."""
    try:
        import netCDF4 as nc
    except ImportError:
        return None
    try:
        ds = nc.Dataset(path, "r")
    except OSError:
        return None
    try:
        hits = {"mean": None, "ping": None, "any": None}
        lat = lon = None

        def find(group, in_sonar=False):
            nonlocal lat, lon
            here = in_sonar or group.path.startswith("/Sonar")
            if "mean_time" in group.variables and len(group.variables["mean_time"]):
                hits["mean"] = hits["mean"] or group.variables["mean_time"]
            if here and "ping_time" in group.variables and len(group.variables["ping_time"]):
                hits["ping"] = hits["ping"] or group.variables["ping_time"]
            if "time" in group.variables and len(group.variables["time"]):
                hits["any"] = hits["any"] or group.variables["time"]
            if "mean_platform_latitude" in group.variables and lat is None:
                lat = group.variables["mean_platform_latitude"]
            if "mean_platform_longitude" in group.variables and lon is None:
                lon = group.variables["mean_platform_longitude"]
            for sub in group.groups.values():
                find(sub, here)

        find(ds)
        tvar = hits["mean"] or hits["ping"] or hits["any"]
        if tvar is None:
            return None
        t0, t1 = _ns1601_to_dt(tvar[0]), _ns1601_to_dt(tvar[-1])
        la = float(lat[0]) if lat is not None and len(lat) else None
        lo = float(lon[0]) if lon is not None and len(lon) else None
        return t0, t1, len(tvar), la, lo
    except (OSError, RuntimeError, KeyError, IndexError, ValueError):
        # truncated / mid-write file (e.g. live remote share): fall back to
        # filename-derived time in scan()
        return None
    finally:
        ds.close()


def peek_raw_time(path: str):
    """``(start, end, None, None, None)`` from a ``.raw``'s first & last datagram headers."""
    try:
        size = os.path.getsize(path)
        with open(path, "rb") as fh:
            def dt_at(off):
                fh.seek(off)
                hdr = fh.read(12)                 # 4-char type + int32 low + int32 high
                if len(hdr) < 12:
                    return None
                low, high = struct.unpack_from("<ii", hdr, 4)
                return _filetime_to_dt(low, high)

            t0 = dt_at(4)                          # payload after the leading length
            fh.seek(size - 4)
            tail = fh.read(4)
            t1 = None
            if len(tail) == 4:
                last_len = struct.unpack("<i", tail)[0]
                if 0 < last_len < size:
                    t1 = dt_at(size - 4 - last_len)
            return (t0, t1, None, None, None) if (t0 or t1) else None
    except (OSError, struct.error):
        return None


def collect(paths) -> list[str]:
    """Expand dirs (recursive ``*.nc``/``*.raw``) and globs to a sorted file list."""
    files: list[str] = []
    for p in (paths if isinstance(paths, (list, tuple)) else [paths]):
        if os.path.isdir(p):
            files += glob.glob(os.path.join(p, "**", "*.nc"), recursive=True)
            files += glob.glob(os.path.join(p, "**", "*.raw"), recursive=True)
        else:
            files += glob.glob(p, recursive=True)
    return sorted(set(files))


def scan(paths, *, peek: bool = True) -> list[dict]:
    """Return one row per EK80 file: ``{file,start,end,n,lat,lon,time_source}`` (start-sorted).

    Missing ``.nc``/``.raw`` ends are filled from the next file's start (flagged
    ``name+next``); a span across a logging gap is therefore only an upper bound.
    """
    rows = []
    for f in collect(paths):
        start = start_from_name(f)
        end = n = la = lo = None
        peeked = False
        if peek:
            pk = (peek_nc_time(f) if f.lower().endswith(".nc")
                  else peek_raw_time(f) if f.lower().endswith(".raw") else None)
            if pk:
                t0, t1, n, la, lo = pk
                start = start or t0
                if t1 is not None:
                    end, peeked = t1, True
        rows.append(dict(file=f, start=start, end=end, n=n, lat=la, lon=lo,
                         time_source="peek" if peeked else "name"))
    rows.sort(key=lambda r: (r["start"] or datetime.max.replace(tzinfo=timezone.utc)))
    for i, r in enumerate(rows):
        if r["end"] is None and r["start"] is not None:
            nxt = rows[i + 1]["start"] if i + 1 < len(rows) else None
            r["end"] = nxt
            if nxt is not None:
                r["time_source"] = "name+next"
    return rows


# ----------------------------------------------------------------- cast correlation
def read_casts(index_path: str) -> list[tuple]:
    """``[(station, utc_datetime, lat, lon), ...]`` from a ``.ladcp_archive.json``."""
    import json
    casts = json.loads(open(index_path).read()).get("casts", {})
    out = []
    for label, c in casts.items():
        utc = c.get("utc")
        if not utc:
            continue
        try:
            dt = datetime.fromisoformat(utc.replace("Z", "+00:00"))
        except ValueError:
            continue
        out.append((label, dt if dt.tzinfo else dt.replace(tzinfo=timezone.utc),
                    c.get("lat"), c.get("lon")))
    return sorted(out, key=lambda x: x[1])


def correlate(rows, casts, *, pre_min: float = 20.0, post_min: float = 170.0,
              file_dur_min: float = 8.0) -> dict[str, list[str]]:
    """Map each cast to the EK80 files overlapping its ``[utc-pre, utc+post]`` window.

    Gap-aware: a file's effective end is ``min(next_start, start + file_dur_min)``, so a
    long transit gap does not let a pre-transit file spuriously cover a later cast.
    """
    starts = sorted(r["start"] for r in rows if r["start"])

    def eff_end(start):
        i = bisect.bisect_right(starts, start)
        end = start + timedelta(minutes=file_dur_min)
        return min(end, starts[i]) if i < len(starts) else end

    out: dict[str, list[str]] = {}
    for label, utc, *_ in casts:
        lo = utc - timedelta(minutes=pre_min)
        hi = utc + timedelta(minutes=post_min)
        out[label] = [r["file"] for r in rows
                      if r["start"] and r["start"] <= hi and eff_end(r["start"]) >= lo]
    return out


# ----------------------------------------------------------------- slim extraction
def _ensure_dims(src_v, dst_grp):
    for dname, dlen in zip(src_v.dimensions, src_v.shape, strict=True):
        if dname not in dst_grp.dimensions:
            dst_grp.createDimension(dname, dlen)


def _copy_var(src_v, dst_grp, name):
    import numpy as np
    _ensure_dims(src_v, dst_grp)
    dt = src_v.datatype
    if hasattr(dt, "dtype"):                       # netCDF4 VLType (ragged)
        vlt = dst_grp.createVLType(dt.dtype, f"{name}_t")
        dv = dst_grp.createVariable(name, vlt, src_v.dimensions)
        data = src_v[:]
        fill = np.nan if np.issubdtype(dt.dtype, np.floating) else 0
        for i in range(len(data)):                 # full length kept: depth = d0 + i*vsi
            row = data[i]
            row = row.filled(fill) if hasattr(row, "filled") else np.asarray(row, dt.dtype)
            dv[i] = np.asarray(row, dt.dtype)
    else:
        dv = dst_grp.createVariable(name, src_v.dtype, src_v.dimensions)
        dv[:] = src_v[:]
    for a in src_v.ncattrs():
        try:
            dv.setncattr(a, src_v.getncattr(a))
        except Exception:
            pass


def slim_extract(src_path: str, dst_path: str) -> int:
    """Copy only the ADCP current group of ``src_path`` to ``dst_path``; return bytes written."""
    import netCDF4 as nc
    with nc.Dataset(src_path, "r") as src:
        adcp_path = next((p for p in _ADCP_CANDIDATES if _has_path(src, p + "/Mean_current")),
                         None)
        if adcp_path is None:
            raise KeyError(f"no /Sonar/Beam_group*/ADCP/Mean_current in {src_path}")
        s_adcp = src[adcp_path]
        s_mc = src[adcp_path + "/Mean_current"]
        os.makedirs(os.path.dirname(dst_path) or ".", exist_ok=True)
        with nc.Dataset(dst_path, "w", format="NETCDF4") as dst:
            dst.setncattr("source_file", os.path.basename(src_path))
            dst.setncattr("note", "slim ADCP-current extract of an EK80 SONAR-netCDF4 file")
            g = dst.createGroup(adcp_path)
            mc = g.createGroup("Mean_current")
            for vn in _KEEP_ADCP:
                if vn in s_adcp.variables:
                    _copy_var(s_adcp.variables[vn], g, vn)
            for vn in _KEEP_MEANCURRENT:
                if vn in s_mc.variables:
                    _copy_var(s_mc.variables[vn], mc, vn)
    return os.path.getsize(dst_path)
