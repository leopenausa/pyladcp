"""Read Simrad EK80 150 kHz ADCP-mode currents into a :class:`~.sadcp_vmdas.SadcpDataset`.

PROTOTYPE. The EK80 echosounder run in ADCP/current mode writes ICES SONAR-netCDF4
files whose current product lives in the group
``/Sonar/Beam_group<g>/ADCP/Mean_current`` as **ragged (vlen)** earth-frame velocity
profiles -- one per averaging period (``mean_time``). Each profile is sampled on a fine
``vertical_sample_interval`` (~0.025 m) grid starting at ``depth_first_sample_center``;
invalid cells are NaN. We rebin to a coarse LADCP-comparable grid and expose the same
interface the inverse ``sadcp=`` constraint already consumes (so Figure 9, the
``sadcp_consistency`` metric and ``--sadcpfac`` all work unchanged).

Currents are already absolute (ship motion removed) and in the geographic frame, so no
heading rotation is applied. Coverage is the **upper water column only** (~15-150 m for
150 kHz) -- this is a shallow constraint, most useful where the LADCP near-surface
coverage is weakest (e.g. single-head/down-only casts).

Works on both the full 541 MB originals and the slim ADCP-only extracts
(``ek80_slim_extract.py``) -- identical group layout.
"""
from __future__ import annotations

import glob
import os

import numpy as np

from .sadcp_vmdas import SadcpDataset

# SONAR-netCDF4 time epoch is 1601-01-01 (ns); seconds from there to the unix epoch.
_S_1601_TO_1970 = 11644473600
_ADCP_GROUP_CANDIDATES = [f"/Sonar/Beam_group{g}/ADCP" for g in range(1, 8)]


def _ek80_time_to_dt64(ns_since_1601) -> np.ndarray:
    """uint64 ns-since-1601 -> datetime64[ns] UTC (1601 is out of ns range, so offset)."""
    v = np.asarray(ns_since_1601).astype("uint64")
    ns70 = (v - np.uint64(_S_1601_TO_1970 * 1_000_000_000)).astype("int64")
    return np.datetime64("1970-01-01T00:00:00", "ns") + ns70.astype("timedelta64[ns]")


def _find_adcp(ds):
    for p in _ADCP_GROUP_CANDIDATES:
        try:
            g = ds[p]
            if "Mean_current" in g.groups:
                return g, ds[p + "/Mean_current"]
        except (IndexError, KeyError):
            continue
    raise KeyError("no /Sonar/Beam_group*/ADCP/Mean_current group found")


def _vlen_to_2d(var, n_mean: int, fill=np.nan) -> np.ndarray:
    """Read a ragged (mean_time,) vlen var into a padded (n_mean, nmax) float array."""
    data = var[:]
    rows = [np.ma.filled(np.ma.masked_invalid(np.asarray(data[i], "float64")), fill)
            for i in range(n_mean)]
    nmax = max((r.size for r in rows), default=0)
    out = np.full((n_mean, nmax), fill, "float64")
    for i, r in enumerate(rows):
        out[i, :r.size] = r
    return out


def read_ek80_file(path, *, bin_m: float = 8.0, z_max: float = 250.0,
                   quality_min: float = 0.0, min_samples: int = 5):
    """Read one EK80 file -> (time[t], lat[t], lon[t], depth[z], u[z,t], v[z,t]).

    Rebins the fine native profiles onto ``0..z_max`` in ``bin_m`` bins, averaging finite
    cells with ``quality >= quality_min``; bins with < ``min_samples`` cells are NaN.
    """
    import netCDF4 as nc

    with nc.Dataset(path, "r") as ds:
        g, mc = _find_adcp(ds)
        t = _ek80_time_to_dt64(mc["mean_time"][:])
        nt = t.size
        east = _vlen_to_2d(mc["current_velocity_geographical_east"], nt)
        north = _vlen_to_2d(mc["current_velocity_geographical_north"], nt)
        qual = (_vlen_to_2d(mc["quality"], nt, fill=0.0)
                if "quality" in mc.variables else np.full_like(east, np.inf))
        lat = np.ma.filled(mc["mean_platform_latitude"][:], np.nan).astype("float64")
        lon = np.ma.filled(mc["mean_platform_longitude"][:], np.nan).astype("float64")
        vsi = float(np.nanmedian(np.ma.filled(g["vertical_sample_interval"][:], np.nan)))
        d0 = float(np.nanmedian(np.ma.filled(g["depth_first_sample_center"][:], np.nan)))

    nmax = east.shape[1]
    depth_native = d0 + np.arange(nmax) * vsi            # index -> depth
    bad = ~np.isfinite(east) | ~np.isfinite(north) | (qual < quality_min)
    east[bad] = np.nan
    north[bad] = np.nan

    edges = np.arange(0.0, z_max + bin_m, bin_m)
    zc = 0.5 * (edges[:-1] + edges[1:])
    nz = zc.size
    which = np.digitize(depth_native, edges) - 1        # bin index per native sample
    valid = (which >= 0) & (which < nz)

    u = np.full((nz, nt), np.nan)
    v = np.full((nz, nt), np.nan)
    for k in range(nz):
        cols = valid & (which == k)
        if not cols.any():
            continue
        e_k = east[:, cols]
        n_k = north[:, cols]
        cnt = np.isfinite(e_k).sum(axis=1)
        enough = cnt >= min_samples
        if not enough.any():                         # all bins empty -> leave NaN
            continue
        with np.errstate(invalid="ignore"):
            u[k, enough] = np.nanmean(e_k[enough], axis=1)
            v[k, enough] = np.nanmean(n_k[enough], axis=1)
    return t, lat, lon, zc, u, v


def read_ek80(source, *, bin_m: float = 8.0, z_max: float = 250.0,
              quality_min: float = 0.0, min_samples: int = 5,
              transducer_depth: float = 5.0) -> SadcpDataset:
    """Read EK80 ADCP currents (a file, dir, or glob) into a SadcpDataset (time-sorted)."""
    files = _collect(source)
    if not files:
        raise FileNotFoundError(f"no EK80 .nc found under {source!r}")

    depth_ref = None
    times, lats, lons, us, vs = [], [], [], [], []
    for f in files:
        t, la, lo, zc, u, v = read_ek80_file(
            f, bin_m=bin_m, z_max=z_max, quality_min=quality_min, min_samples=min_samples)
        if t.size == 0:
            continue
        depth_ref = zc if depth_ref is None else depth_ref
        times.append(t); lats.append(la); lons.append(lo)
        us.append(u); vs.append(v)
    if not times:
        raise ValueError(f"EK80 source {source!r} yielded no usable ensembles")

    time = np.concatenate(times)
    lat = np.concatenate(lats)
    lon = np.concatenate(lons)
    u = np.concatenate(us, axis=1)
    v = np.concatenate(vs, axis=1)
    order = np.argsort(time)
    return SadcpDataset(
        time=time[order], lat=lat[order], lon=lon[order],
        depth=depth_ref.astype("float64"),
        u=u[:, order], v=v[:, order],
        freq_khz=150, transducer_depth=float(transducer_depth),
        file_type="EK80", source=str(source), n_files=len(files))


def _collect(source) -> list[str]:
    if isinstance(source, (list, tuple)):
        out = []
        for s in source:
            out += _collect(s)
        return sorted(set(out))
    p = str(source)
    if os.path.isdir(p):
        return sorted(glob.glob(os.path.join(p, "**", "*.nc"), recursive=True))
    if any(ch in p for ch in "*?["):
        return sorted(glob.glob(p, recursive=True))
    return [p] if os.path.exists(p) else []


if __name__ == "__main__":          # quick manual check
    import sys
    ds = read_ek80(sys.argv[1])
    print(f"EK80 SadcpDataset: {ds.n_ens} ens, depth {ds.depth[0]:.0f}-{ds.depth[-1]:.0f} m "
          f"({ds.depth.size} bins), {ds.n_files} file(s)")
    print(f"time {ds.time[0]} .. {ds.time[-1]}")
    with np.errstate(invalid="ignore"):
        print(f"u median {np.nanmedian(ds.u):+.3f}  v median {np.nanmedian(ds.v):+.3f} m/s")
        cov = np.isfinite(ds.u).mean(axis=1)
    for z, c in zip(ds.depth, cov, strict=True):
        if c > 0:
            print(f"  z={z:6.1f} m  coverage={c*100:4.0f}%  "
                  f"u={np.nanmedian(ds.u[int(z//8)]):+.3f}")
