"""Instrument depth from the CTD pressure time-series, time-matched to the LADCP.

Legacy ``loadctd``/``getdpth`` align the CTD clock to the LADCP by cross-correlating
the CTD-derived depth with the depth obtained by integrating the LADCP vertical
velocity, then use the CTD depth (``p.ctddepth=1``) as the package depth. We reproduce
that: integrate w for a relative depth, find the best time lag against CTD depth, and
map CTD depth onto the LADCP ensemble times.

Sign convention follows the legacy ``d.z`` (negative = below surface).
"""

from __future__ import annotations

from dataclasses import dataclass

import gsw
import numpy as np

from .prepare import Ensembles


@dataclass
class DepthSolution:
    z: np.ndarray              # [t] instrument depth, negative down (legacy d.z)
    izm: np.ndarray            # [nbin, t] depth of every measurement, negative down
    max_depth: float           # positive metres
    ctd_t: np.ndarray          # [t] CTD temperature at ensembles
    ctd_s: np.ndarray          # [t] CTD salinity at ensembles
    lag_seconds: float
    in_water: np.ndarray       # [t] bool mask of the in-water cast window


def integrate_w_depth(ens: Ensembles) -> np.ndarray:
    """Relative depth (positive down) from integrating mean vertical velocity."""
    izr = ens.izd[:6] if ens.izd.size >= 6 else ens.izd
    wm = np.nanmedian(ens.rw[izr], axis=0)
    wm = np.where(np.isfinite(wm), wm, 0.0)
    dt = _dt_seconds(ens.time)
    # During descent the package moves down, so water appears to move UP relative to
    # the instrument (w>0). Depth (positive down) = cumsum(w·dt)  (legacy getdpth).
    return np.cumsum(wm * dt)


def solve_depth(ens: Ensembles, ctd) -> DepthSolution:
    tl = _seconds(ens.time)
    ladcp_depth = integrate_w_depth(ens)
    ladcp_depth -= ladcp_depth[0]

    ctd_depth = -gsw.z_from_p(ctd.pressure, np.nanmean(ctd.lat))   # positive down
    ctd_t = ctd.time_elapsed_s.astype(float)

    lag = _best_lag(tl, ladcp_depth, ctd_t, ctd_depth)

    # map CTD depth/T/S onto LADCP ensemble times
    z_pos = np.interp(tl, ctd_t + lag, ctd_depth, left=np.nan, right=np.nan)
    tC = np.interp(tl, ctd_t + lag, ctd.temperature, left=np.nan, right=np.nan)
    sP = np.interp(tl, ctd_t + lag, ctd.salinity, left=np.nan, right=np.nan)

    # fall back to integrated-w depth where CTD does not cover
    bad = ~np.isfinite(z_pos)
    z_pos[bad] = ladcp_depth[bad]
    z = -z_pos                                  # negative down (legacy d.z)

    # combined measurement-depth array: offsets [flip(zu) (+up), -zd (down)]
    off = np.concatenate([np.flip(ens.zu), -ens.zd]) if ens.zu.size else -ens.zd
    izm = z[None, :] + off[:, None]

    max_depth = float(np.nanmax(z_pos))
    in_water = z_pos > 10.0
    return DepthSolution(z=z, izm=izm, max_depth=max_depth, ctd_t=tC, ctd_s=sP,
                         lag_seconds=float(lag), in_water=in_water)


def _best_lag(tl, ld, ct, cd) -> float:
    """Coarse argmax-of-depth alignment, refined by local cross-correlation (seconds)."""
    # coarse: align the time of maximum depth
    lag0 = tl[int(np.nanargmax(ld))] - ct[int(np.nanargmax(cd))]
    # refine on a common 1 s grid around lag0
    grid = np.arange(tl[0], tl[-1], 1.0)
    ld_g = np.interp(grid, tl, ld)
    best, best_c = lag0, -np.inf
    for dlag in np.arange(-90, 91, 1.0):
        cd_g = np.interp(grid, ct + lag0 + dlag, cd, left=np.nan, right=np.nan)
        m = np.isfinite(cd_g)
        if m.sum() < 10:
            continue
        c = np.corrcoef(ld_g[m], cd_g[m])[0, 1]
        if c > best_c:
            best_c, best = c, lag0 + dlag
    return best


def _seconds(t):
    return (t - t[0]) / np.timedelta64(1, "s")


def _dt_seconds(t):
    s = _seconds(t)
    d = np.diff(s)
    return np.concatenate([[d[0]], (d[:-1] + d[1:]) / 2, [d[-1]]]) if len(d) > 1 else np.array([1.0])
