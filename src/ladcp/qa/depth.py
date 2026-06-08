"""CTD <-> LADCP time synchronization and package depth (schematic's central box).

The LADCP instrument clock is unreliable (not GPS-set), so absolute PD0 timestamps
cannot be trusted to align the two instruments. Instead — exactly as the legacy
``loadctd``/``getdpthi`` chain does — we synchronize on *motion*: the CTD vertical
velocity (d depth / dt) and the LADCP reference vertical velocity are cross-correlated to
find the time offset, then the CTD depth is interpolated onto the (shifted) ADCP ping
times to give the package depth ``z`` per ensemble.

The alignment is two-stage. The ADCP commonly records for far longer than the cast (it
pings on deck before deployment and after recovery), so its valid in-water window can sit
thousands of pings into the file — a clock offset of minutes. :func:`_coarse_offset` first
slides the (short) CTD vertical-velocity window across the (long) ADCP one over the whole
record to locate the cast (a normalized cross-correlation); :func:`bestlag` then refines
the residual integer lag on the coarse-aligned grid. The single-stage ``bestlag`` the
pipeline used before could only reach a ``±nlag`` offset (and structurally no more than
~half the CTD length), so a cast buried deep in the recording was silently mis-mapped onto
the on-deck pings — the in-water depth window then contained no valid velocity and every
super-ensemble collapsed to NaN (FDCCC1_001).

Validated against MORIA-80: maxdepth 1072.7 m (golden ``p.maxdepth`` 1072.68) and a
synchronization correlation of ~0.97 (golden 0.984); the coarse stage returns the same
431 s offset the old single-stage search converged to, so that result is unchanged.
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
    lag: int                     # total ADCP-minus-CTD elapsed time offset [s] (coarse + fine)
    corr: float                  # correlation of CTD vs LADCP vertical velocity at that offset
    z_on_ping: np.ndarray        # [nens] package depth per ADCP ping [m, +down]
    maxdepth: float              # deepest package depth [m]
    ctd_maxdepth: float          # deepest CTD depth [m]
    coarse_score: float = np.nan  # peak normalized cross-correlation of the coarse localization
                                  # (low -> the cast could not be confidently located; WARN)


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


def _coarse_offset(tad: np.ndarray, w_ad: np.ndarray,
                   t_ctd: np.ndarray, w_ctd: np.ndarray,
                   *, dt: float = 1.0) -> tuple[float, float]:
    """Locate the (short) CTD cast within the (possibly long) ADCP record.

    Returns ``(offset_s, score)`` where ``offset_s`` is the time offset such that CTD elapsed
    time ``t`` corresponds to ADCP elapsed time ``t + offset_s``, found by sliding the CTD
    vertical-velocity window across the ADCP one over the *whole* record and maximizing the
    normalized cross-correlation (the descent-then-ascent W signature gives a sharp peak).
    ``score`` is the peak correlation (in ``[-1, 1]``); a low value means the cast could not be
    confidently located. Robust to the ADCP pinging on deck for far longer than the cast and to
    CTD/ADCP clock offsets that exceed :func:`bestlag`'s reach. Returns ``(0, 0)`` when either
    series is too short to localize.
    """
    tad = np.asarray(tad, float)
    t_ctd = np.asarray(t_ctd, float)
    if (tad.size < 2 or t_ctd.size < 2
            or not np.isfinite(tad[-1]) or not np.isfinite(t_ctd[-1])):
        return 0.0, 0.0
    g_ad = np.arange(0.0, float(tad[-1]) + dt, dt)
    wa = np.interp(g_ad, tad, np.where(np.isfinite(w_ad), w_ad, 0.0), left=0.0, right=0.0)
    g_ct = np.arange(0.0, float(t_ctd[-1]) + dt, dt)
    wc = np.interp(g_ct, t_ctd, np.where(np.isfinite(w_ctd), w_ctd, 0.0), left=0.0, right=0.0)
    m, n = wc.size, wa.size
    if m < 5 or n < m:
        return 0.0, 0.0
    wc0 = wc - wc.mean()
    nc = float(np.sqrt((wc0 ** 2).sum()))
    if nc <= 0.0:
        return 0.0, 0.0
    # sliding-window mean/variance of the ADCP series via cumulative sums (vectorized)
    csum = np.concatenate(([0.0], np.cumsum(wa)))
    csq = np.concatenate(([0.0], np.cumsum(wa * wa)))
    seg_mean = (csum[m:] - csum[:-m]) / m
    seg_var = (csq[m:] - csq[:-m]) - m * seg_mean ** 2          # sum of squared deviations
    cross = np.correlate(wa, wc0, mode="valid")                # sum over window of wa * wc0
    denom = np.sqrt(np.maximum(seg_var, 0.0)) * nc
    score = np.full(denom.shape, -np.inf)
    ok = denom > 0.0
    score[ok] = cross[ok] / denom[ok]
    k = int(np.argmax(score))
    return float(g_ad[k]), float(score[k])


def synchronize(dh: DualHead, ctd: CTDTimeSeries, *, nlag: int = 600) -> SyncResult:
    """Synchronize CTD to LADCP and return the package depth on ping times.

    Coarse-localizes the cast within the full ADCP record (:func:`_coarse_offset`), then
    refines the residual integer lag with :func:`bestlag` on the coarse-aligned CTD grid.
    ``nlag`` caps the fine search; it is further clamped to ``< t_ctd.size/2`` so the
    ``bestlag`` correlation window cannot collapse to its degenerate ``(0, 1.0)`` early exit
    on short CTD series.
    """
    depth = ctd_depth(ctd)
    t_ctd = np.asarray(ctd.time_elapsed_s, float)
    w_ctd = np.gradient(depth, t_ctd)

    w_ad = reference_w(dh)
    tad = (dh.down.time - dh.down.time[0]) / np.timedelta64(1, "s")

    tau, coarse_score = _coarse_offset(tad, w_ad, t_ctd, w_ctd)

    # fine integer refinement on the coarse-aligned grid (sub-coarse / sign cleanup)
    w_ad_on_ctd = np.interp(t_ctd + tau, tad,
                            np.where(np.isfinite(w_ad), w_ad, 0.0),
                            left=np.nan, right=np.nan)
    nlag_fine = int(max(2, min(nlag, t_ctd.size // 3)))
    lag, corr = bestlag(w_ctd, w_ad_on_ctd, nlag=nlag_fine)
    offset = float(tau) + float(lag)

    # CTD elapsed time corresponding to each ADCP ping is (tad - offset)
    z_on_ping = np.interp(tad - offset, t_ctd, depth, left=np.nan, right=np.nan)
    return SyncResult(
        lag=int(round(offset)), corr=corr, z_on_ping=z_on_ping,
        maxdepth=float(np.nanmax(z_on_ping)) if np.isfinite(z_on_ping).any() else np.nan,
        ctd_maxdepth=float(np.nanmax(depth)),
        coarse_score=coarse_score,
    )
