"""Ingest VmDAS shipboard-ADCP (SADCP) data for the LADCP inverse constraint.

VmDAS (Teledyne RDI's Vessel-mount Data Acquisition Software) writes averaged
ensembles as PD0 files with extra data types:

* ``.STA`` -- short-term averages (here ~2 min); finer time resolution, used by default.
* ``.LTA`` -- long-term averages; coarser.

The velocity data type (0x0100) holds the **water-track** earth velocity, i.e. the
ocean velocity *relative to the moving ship*; underway this is dominated by the ship's
~5 m/s motion. The VmDAS navigation block (0x2000) carries the GPS fixes. Absolute
ocean velocity is recovered as ``water_track + ship_velocity``, where the ship velocity
is the time-derivative of the navigated position (validated on MORIA: the ~4.7 m/s
underway signal collapses to ocean scale once added).

This module mirrors the legacy LDEO_IX SADCP path (``mkSADCP.m`` builds an absolute
``u_sadcp(z,t)`` time series; ``loadsadcp.m`` windows it to the cast). :func:`ingest_dir`
is the ``mkSADCP`` step (parse a folder once, cache the small time series); the
:class:`SadcpDataset` cache is reused without re-parsing the raw tree unless ``force``;
:func:`extract_profile` is the ``loadsadcp`` step (window + average to ``svel``).

References: Teledyne RDI "VmDAS User's Guide" (binary output, navigation data block);
LDEO_IX ``loadsadcp.m`` / ``getinv.m`` ``lainsadcp``.
"""

from __future__ import annotations

import struct
from dataclasses import dataclass
from pathlib import Path

import numpy as np

from .pd0 import _ensemble_offsets, _find_type, read_pd0

# VmDAS navigation data block (per-ensemble GPS fixes). Latitude/longitude are stored
# as signed 32-bit "binary angle measure": degrees = raw * 180 / 2**31. Offsets are
# relative to the data-type ID and were verified against the known MORIA on-station
# position (first fix at +14/+18, last fix at +26/+30).
ID_VMDAS_NAV = 0x2000
_BAM = 180.0 / 2**31
_NAV_FIRST_LAT = 14
_NAV_FIRST_LON = 18
_NAV_LAST_LAT = 26
_NAV_LAST_LON = 30

_M_PER_DEG_LAT = 111_320.0
CACHE_NAME = "sadcp_cache.npz"


@dataclass
class SadcpDataset:
    """Absolute ocean-velocity time series from one shipboard-ADCP instrument.

    ``u``/``v`` are east/north ocean velocity in the **true** (geographic) frame, the
    frame :func:`ladcp.qa.inverse.compute_velocity_full` expects for its ``sadcp=``
    constraint. ``depth`` is positive-down below the sea surface.
    """

    time: np.ndarray            # [t] datetime64[ns] UTC (sorted)
    lat: np.ndarray             # [t] deg N
    lon: np.ndarray             # [t] deg E
    depth: np.ndarray           # [z] m, positive down
    u: np.ndarray               # [z, t] absolute ocean east, m/s
    v: np.ndarray               # [z, t] absolute ocean north, m/s
    freq_khz: int
    transducer_depth: float
    file_type: str              # "STA"/"LTA" (raw VmDAS) or "CODAS" (sadcp_codas)
    source: str                 # ingested folder
    n_files: int

    @property
    def n_ens(self) -> int:
        return int(self.time.size)


def _read_nav(buf: bytes, ensembles, transducer_depth: float):
    """Per-ensemble ship position (mean of first/last GPS fix) from the 0x2000 block."""
    n = len(ensembles)
    lat = np.full(n, np.nan)
    lon = np.full(n, np.nan)
    for i, (pos, _ndt, offs) in enumerate(ensembles):
        b = _find_type(buf, pos, offs, ID_VMDAS_NAV)
        if b is None:
            continue
        la1 = struct.unpack_from("<i", buf, b + _NAV_FIRST_LAT)[0] * _BAM
        lo1 = struct.unpack_from("<i", buf, b + _NAV_FIRST_LON)[0] * _BAM
        la2 = struct.unpack_from("<i", buf, b + _NAV_LAST_LAT)[0] * _BAM
        lo2 = struct.unpack_from("<i", buf, b + _NAV_LAST_LON)[0] * _BAM
        lat[i] = 0.5 * (la1 + la2)
        lon[i] = 0.5 * (lo1 + lo2)
    return lat, lon


def read_vmdas_file(path: str, *, transducer_depth: float = 5.0,
                    pctgood_min: float = 25.0):
    """Parse one VmDAS STA/LTA file.

    Returns ``(time, lat, lon, depth, u_water, v_water, freq_khz)`` where ``u_water``/
    ``v_water`` are the *ship-relative* earth velocities ``[z, t]`` (absolute velocity is
    formed in :func:`ingest_dir`). ``depth`` is positive-down below the surface, built
    from the fixed-leader cell geometry plus ``transducer_depth`` (the instrument's depth
    below the waterline, ~5 m for a hull-mounted Ocean Surveyor).

    Cells whose four-beam percent-good is below ``pctgood_min`` are blanked: unlike the
    CODAS-processed series the legacy assumed, raw VmDAS writes unscreened noise in
    out-of-range bins, which this removes before any averaging.
    """
    with open(path, "rb") as fh:
        buf = fh.read()
    ensembles = list(_ensemble_offsets(buf))
    if not ensembles:
        raise ValueError(f"no valid VmDAS ensembles in {path}")
    raw = read_pd0(path)
    time = np.asarray(raw.time)
    # read_pd0 stores earth velocity as the 4 "beam" slots: 0=east, 1=north, 2=up, 3=err
    u_water = raw.vel[0].copy()
    v_water = raw.vel[1].copy()
    if raw.pct_good is not None and pctgood_min > 0:
        # earth-frame percent-good field 3 = fraction of good 4-beam transforms
        bad = raw.pct_good[3] < pctgood_min
        u_water[bad] = np.nan
        v_water[bad] = np.nan
    nz = u_water.shape[0]
    # cell-centre depth below surface: transducer depth + range to bin 1 + k * cell size
    dist1 = float(raw.meta.get("dist_first_m", raw.blank_m + raw.cell_m))
    depth = transducer_depth + dist1 + np.arange(nz) * raw.cell_m
    lat, lon = _read_nav(buf, ensembles, transducer_depth)
    return time, lat, lon, depth, u_water, v_water, raw.freq_khz


def _ship_velocity(time: np.ndarray, lat: np.ndarray, lon: np.ndarray):
    """East/north ship velocity [m/s] from the time-derivative of the GPS track.

    Central difference of the navigated position over the (monotonic, contiguous) per-file
    time vector. On station the ship is ~stationary so this is ~0; underway it recovers the
    vessel motion that must be added back to the water-track velocity.
    """
    ts = (time - time[0]) / np.timedelta64(1, "s")
    good = np.isfinite(ts) & np.isfinite(lat) & np.isfinite(lon)
    us = np.full(time.size, np.nan)
    vs = np.full(time.size, np.nan)
    if good.sum() < 2:
        return us, vs
    mlat = np.nanmean(lat)
    m_per_deg_lon = _M_PER_DEG_LAT * np.cos(np.radians(mlat))
    x = lon * m_per_deg_lon
    y = lat * _M_PER_DEG_LAT
    # np.gradient needs strictly increasing coordinates; restrict to the good, sorted set
    idx = np.where(good)[0]
    if idx.size >= 2:
        usg = np.gradient(x[idx], ts[idx])
        vsg = np.gradient(y[idx], ts[idx])
        us[idx] = usg
        vs[idx] = vsg
    return us, vs


def ingest_dir(folder: str | Path, *, file_type: str = "STA",
               transducer_depth: float = 5.0, pctgood_min: float = 25.0) -> SadcpDataset:
    """Parse every ``*.<file_type>`` in ``folder`` into one absolute-velocity dataset.

    This is the expensive step (reads the raw tree). Concatenates all files, recovers
    absolute ocean velocity per file (``water_track + ship_velocity``), and sorts by time.
    Cache the result with :func:`save_cache` / :func:`load_or_ingest` to avoid repeating it.
    """
    folder = Path(folder)
    ext = file_type.upper()
    files = sorted(folder.glob(f"*.{ext}")) + sorted(folder.glob(f"*.{ext.lower()}"))
    files = sorted(set(files))
    if not files:
        raise FileNotFoundError(f"no .{ext} files under {folder}")

    depth0 = None
    freq0 = None
    times, lats, lons, us, vs = [], [], [], [], []
    for f in files:
        time, lat, lon, depth, uw, vw, freq = read_vmdas_file(
            str(f), transducer_depth=transducer_depth, pctgood_min=pctgood_min)
        if time.size == 0:
            continue
        if depth0 is None:
            depth0, freq0 = depth, freq
        elif depth.shape != depth0.shape:
            # different cell geometry -> a different instrument/config; keep this folder
            # to one instrument rather than silently mixing grids.
            raise ValueError(
                f"{f.name}: {depth.size} cells but {depth0.size} expected; "
                "point the ingester at a single-instrument folder")
        ush, vsh = _ship_velocity(time, lat, lon)
        us.append(uw + ush[None, :])        # absolute = water-track + ship velocity
        vs.append(vw + vsh[None, :])
        times.append(time)
        lats.append(lat)
        lons.append(lon)

    if not times:
        raise ValueError(f"no ensembles parsed from {folder}")
    time = np.concatenate(times)
    # an acquisition folder can hold the same ensembles twice (e.g. a compiled
    # whole-cruise STA next to the per-deployment files it was built from, as on
    # cruise 2) -- identical timestamps are the same ping average, keep one
    _, order = np.unique(time, return_index=True)   # time-sorted, first occurrence wins
    return SadcpDataset(
        time=time[order],
        lat=np.concatenate(lats)[order],
        lon=np.concatenate(lons)[order],
        depth=depth0,
        u=np.concatenate(us, axis=1)[:, order],
        v=np.concatenate(vs, axis=1)[:, order],
        freq_khz=int(freq0),
        transducer_depth=float(transducer_depth),
        file_type=ext,
        source=str(folder),
        n_files=len(files),
    )


def save_cache(ds: SadcpDataset, path: str | Path) -> None:
    """Write a dataset to a compact ``.npz`` cache."""
    np.savez_compressed(
        path,
        time=ds.time.astype("datetime64[ns]").astype("int64"),
        lat=ds.lat, lon=ds.lon, depth=ds.depth, u=ds.u, v=ds.v,
        freq_khz=ds.freq_khz, transducer_depth=ds.transducer_depth,
        file_type=ds.file_type, source=ds.source, n_files=ds.n_files,
    )


def load_cache(path: str | Path) -> SadcpDataset:
    """Read a dataset back from a ``.npz`` cache written by :func:`save_cache`."""
    z = np.load(path, allow_pickle=False)
    return SadcpDataset(
        time=z["time"].astype("datetime64[ns]"),
        lat=z["lat"], lon=z["lon"], depth=z["depth"], u=z["u"], v=z["v"],
        freq_khz=int(z["freq_khz"]), transducer_depth=float(z["transducer_depth"]),
        file_type=str(z["file_type"]), source=str(z["source"]), n_files=int(z["n_files"]),
    )


def load_or_ingest(folder: str | Path, *, cache: str | Path | None = None,
                   force: bool = False, file_type: str = "STA",
                   transducer_depth: float = 5.0,
                   pctgood_min: float = 25.0) -> SadcpDataset:
    """Return the dataset for ``folder``, building (and caching) it only when needed.

    Loads the cache if present and ``force`` is false; otherwise parses the raw tree and
    writes the cache. ``cache`` defaults to ``<folder>/sadcp_cache.npz``. This is the
    "parse once, reuse later" entry point.
    """
    folder = Path(folder)
    cache_path = Path(cache) if cache is not None else folder / CACHE_NAME
    if cache_path.exists() and not force:
        return load_cache(cache_path)
    ds = ingest_dir(folder, file_type=file_type, transducer_depth=transducer_depth,
                    pctgood_min=pctgood_min)
    try:
        save_cache(ds, cache_path)
    except OSError:
        pass        # read-only location: still return the freshly ingested dataset
    return ds


def extract_profile(ds: SadcpDataset, *, time_start, time_end,
                    lat: float | None = None, lon: float | None = None,
                    dtok_min: float = 0.0, maxdepth: float = np.inf,
                    zbottom: float = np.nan, pos_tol_deg: float = 0.1,
                    min_samples: int = 3, verr_max: float = 1.0):
    """Window the dataset to one cast and average to ``svel = [z, u, v, verr]``.

    Port of LDEO_IX ``loadsadcp.m``: select ensembles in ``[start - dtok, end + dtok]``;
    reject (return ``None``) if their mean position is more than ``pos_tol_deg`` from the
    LADCP fix (longitude tolerance widened by ``1/cos(lat)``); time-average u/v per depth;
    set ``verr`` to the across-ensemble standard deviation (floor 0.1 m/s) scaled by
    ``max(n)/n`` so sparsely sampled depths weigh less; keep depths above ``maxdepth`` and
    ``zbottom - 30``. Depths sampled in fewer than ``min_samples`` ensembles, or whose scaled
    ``verr`` exceeds ``verr_max`` (effectively unconstrained out-of-range noise the legacy's
    CODAS step would have screened), are dropped. Returns an ``(n, 4)`` array in the true
    frame, or ``None`` if no usable data. ``ds.u``/``ds.v`` are already absolute ocean
    velocity.
    """
    import warnings

    t0 = np.datetime64(time_start) - np.timedelta64(int(dtok_min * 60), "s")
    t1 = np.datetime64(time_end) + np.timedelta64(int(dtok_min * 60), "s")
    sel = (ds.time >= t0) & (ds.time <= t1)
    if sel.sum() < 1:
        return None

    if lat is not None and lon is not None:
        mlat = float(np.nanmean(ds.lat[sel]))
        mlon = float(np.nanmean(ds.lon[sel]))
        lon_tol = pos_tol_deg / max(np.cos(np.radians(lat)), 1e-6)
        if abs(mlon - lon) > lon_tol or abs(mlat - lat) > pos_tol_deg:
            return None        # SADCP track too far from the LADCP station

    u = ds.u[:, sel]
    v = ds.v[:, sel]
    n = np.sum(np.isfinite(u) & np.isfinite(v), axis=1)        # samples per depth
    enough = n >= min_samples
    if not enough.any():
        return None

    with warnings.catch_warnings():
        warnings.simplefilter("ignore", RuntimeWarning)        # empty/1-sample slices
        umean = np.nanmean(u, axis=1)
        vmean = np.nanmean(v, axis=1)
        verr = np.nanstd(u, axis=1, ddof=1) + np.nanstd(v, axis=1, ddof=1)
    # floor a zero / near-zero / non-finite scatter to 0.1 m/s (legacy loadsadcp floor),
    # so a degenerate verr cannot blow up to near-infinite constraint weight
    verr = np.where(np.isfinite(verr) & (verr > 1e-6), verr, 0.1)
    nmax = n[enough].max()
    with np.errstate(divide="ignore", invalid="ignore"):
        verr = verr * nmax / np.where(n > 0, n, np.nan)

    z = ds.depth
    ok = enough & np.isfinite(umean + vmean) & (z < maxdepth) & (verr <= verr_max)
    if np.isfinite(zbottom):
        ok &= z < (zbottom - 30.0)
    if not ok.any():
        return None
    return np.column_stack([z[ok], umean[ok], vmean[ok], verr[ok]])
