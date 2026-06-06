"""CTD <-> LADCP time synchronization and package depth (schematic's central box).

The LADCP instrument clock is unreliable (not GPS-set), so absolute PD0 timestamps
cannot be trusted to align the two instruments. Instead — exactly as the legacy
``loadctd``/``getdpthi`` chain does — we synchronize on *motion*: the CTD vertical
velocity (d depth / dt) and the LADCP reference vertical velocity are cross-correlated
with :func:`bestlag` to find the integer lag, then the CTD depth is interpolated onto the
(shifted) ADCP ping times to give the package depth ``z`` per ensemble.

Validated against MORIA-80: maxdepth 1072.7 m (golden ``p.maxdepth`` 1072.68) and a
synchronization correlation of ~0.97 (golden 0.984).
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np

from ..models import CTDTimeSeries
from .bestlag import bestlag
from .ingest import DualHead

_W_COMP = 2          # earth coord component 3 = vertical velocity


def ctd_depth(ctd: CTDTimeSeries) -> np.ndarray:
    """Depth [m, positive down] from CTD pressure via TEOS-10 (``gsw.z_from_p``)."""
    import gsw
    lat = float(np.nanmedian(ctd.lat))
    return -gsw.z_from_p(ctd.pressure, lat)


@dataclass
class SyncResult:
    lag: int                     # ADCP-elapsed minus CTD-elapsed [scans/seconds]
    corr: float                  # correlation of CTD vs LADCP vertical velocity
    z_on_ping: np.ndarray        # [nens] package depth per ADCP ping [m, +down]
    maxdepth: float              # deepest package depth [m]
    ctd_maxdepth: float          # deepest CTD depth [m]


def water_window(z_on_ping: np.ndarray, threshold: float = 10.0) -> tuple[int, int]:
    """First and last ping index with package depth below ``threshold`` (in-water span)."""
    inw = np.where(np.isfinite(z_on_ping) & (z_on_ping > threshold))[0]
    if inw.size == 0:
        return 0, z_on_ping.size - 1
    return int(inw[0]), int(inw[-1])


def reference_w(dh: DualHead) -> np.ndarray:
    """LADCP reference vertical velocity per ensemble (mean of earth-w over bins)."""
    import warnings
    w = dh.down.vel[_W_COMP]
    with warnings.catch_warnings():
        warnings.simplefilter("ignore", RuntimeWarning)   # all-NaN columns -> NaN
        return np.nanmean(w, axis=0)


def synchronize(dh: DualHead, ctd: CTDTimeSeries, *, nlag: int = 600) -> SyncResult:
    """Synchronize CTD to LADCP and return the package depth on ping times."""
    depth = ctd_depth(ctd)
    w_ctd = np.gradient(depth, ctd.time_elapsed_s)

    w_ad = reference_w(dh)
    tad = (dh.down.time - dh.down.time[0]) / np.timedelta64(1, "s")
    w_ad_on_ctd = np.interp(ctd.time_elapsed_s, tad,
                            np.where(np.isfinite(w_ad), w_ad, 0.0),
                            left=np.nan, right=np.nan)

    lag, corr = bestlag(w_ctd, w_ad_on_ctd, nlag=nlag)
    # CTD elapsed time corresponding to each ADCP ping is (tad - lag)
    z_on_ping = np.interp(tad - lag, ctd.time_elapsed_s, depth,
                          left=np.nan, right=np.nan)
    return SyncResult(
        lag=lag, corr=corr, z_on_ping=z_on_ping,
        maxdepth=float(np.nanmax(z_on_ping)),
        ctd_maxdepth=float(np.nanmax(depth)),
    )
