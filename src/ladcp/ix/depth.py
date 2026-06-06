"""Depth from integrated vertical velocity (getdpth.m, W-integration core).

This module reproduces the LDEO_IX depth-from-W machinery used in ``getdpth.m``:
``getmeanw`` (reference-bin mean vertical velocity, smoothed and gap-filled) and the
first-sweep down-trace / up-trace depth integrals with sound-speed correction. Those two
max-depths are printed in the golden processing log (``maxdepth from down-trace`` /
``up-trace``) and form a clean, ADCP-only validation of the earth-velocity ``w`` chain
(independent of the CTD time-lag join, which getdpth uses only when ``ctddepth==1``).
"""

from __future__ import annotations

import numpy as np

from ..io.ctd_seabird import SeabirdCast, p2z
from .ingest import LADCPData, _quiet_nan


def centred_dt(time: np.ndarray) -> np.ndarray:
    """Centred per-ensemble time step in seconds (legacy ``mean([0,dt;dt,0])``)."""
    t = time.astype("datetime64[ns]").astype(np.int64) / 1e9
    d = np.diff(t)
    return 0.5 * (np.concatenate([[0.0], d]) + np.concatenate([d, [0.0]]))


def get_mean_w(d: LADCPData, wizr: np.ndarray | None = None) -> np.ndarray:
    """Reference-bin mean vertical velocity ``wm`` (getmeanw subfunction).

    ``wizr`` are the combined-row reference bins (default: first 5 down + first 5 up bins,
    matching loadrdi's ``izr``). Returns smoothed, gap-filled ``wm`` ``[nens]``.
    """
    if wizr is None:
        wizr = d.izd[:5]
        if d.izu.size:
            wizr = np.concatenate([wizr, d.izu[:5]])
    sub = d.rw[wizr, :]
    with _quiet_nan():
        wm1 = np.nanmedian(sub, axis=0)
    finite = np.isfinite(sub).sum(axis=0)
    wm1[finite < 2] = np.nan

    # 4-point centred smoothing (weights 1,2,1 over [i-1,i,i,i+1])
    wm = wm1.copy()
    ii = slice(1, len(wm1) - 1)
    stack = np.vstack([wm1[:-2], wm1[1:-1], wm1[1:-1], wm1[2:]])
    with _quiet_nan():
        wm[ii] = np.nanmean(stack, axis=0)

    # propagate from neighbours up to 3 passes, then zero-fill
    for nn in range(1, 4):
        bad = ~np.isfinite(wm)
        if not bad.any():
            break
        idx = np.where(bad)[0]
        dat = []
        for k in range(1, nn + 1):
            dat.append(wm[np.clip(idx - k, 0, len(wm) - 1)])
            dat.append(wm[np.clip(idx + k, 0, len(wm) - 1)])
        with _quiet_nan():
            wm[idx] = np.nanmean(np.vstack(dat), axis=0)
    wm[~np.isfinite(wm)] = 0.0
    return wm


def _sounds(p_dbar: np.ndarray, t: np.ndarray, s: float = 34.5) -> np.ndarray:
    """Sound speed [m/s] from pressure/temperature/salinity (legacy ``sounds.m``)."""
    P = p_dbar / 10.0
    SR = np.sqrt(abs(s))
    D = 1.727e-3 - 7.8936e-6 * P
    B1 = 7.3637e-5 + 1.7945e-7 * t
    B0 = -1.922e-2 - 4.42e-5 * t
    B = B0 + B1 * P
    A3 = (-3.389e-13 * t + 6.649e-12) * t + 1.100e-10
    A2 = ((7.988e-12 * t - 1.6002e-10) * t + 9.1041e-9) * t - 3.9064e-7
    A1 = (((-2.0122e-10 * t + 1.0507e-8) * t - 6.4885e-8) * t - 1.2580e-5) * t + 9.4742e-5
    A0 = (((-3.21e-8 * t + 2.006e-6) * t + 7.164e-5) * t - 1.262e-2) * t + 1.389
    A = ((A3 * P + A2) * P + A1) * P + A0
    C3 = (-2.3643e-12 * t + 3.8504e-10) * t - 9.7729e-9
    C2 = (((1.0405e-12 * t - 2.5335e-10) * t + 2.5974e-8) * t - 1.7107e-6) * t + 3.1260e-5
    C1 = (((-6.1185e-10 * t + 1.3621e-7) * t - 8.1788e-6) * t + 6.8982e-4) * t + 0.153563
    C0 = ((((3.1464e-9 * t - 1.47800e-6) * t + 3.3420e-4) * t - 5.80852e-2) * t
          + 5.03711) * t + 1402.388
    C = ((C3 * P + C2) * P + C1) * P + C0
    return C + (A + B * SR + D * s) * s


def _press(z: np.ndarray) -> np.ndarray:
    """Pressure [dbar] from depth [m] (legacy ``press.m``)."""
    C1, C2, C3 = 2.398599584e05, 5.753279964e10, 4.833657881e05
    return C1 - np.sqrt(C2 - C3 * z)


def trace_depths(d: LADCPData, soundc: bool = True) -> dict:
    """First-sweep down/up-trace max depths from integrated W (getdpth n==1).

    Returns ``{down_max, up_max, wm}`` where the two max-depths correspond to the golden
    log lines ``maxdepth from down-trace`` / ``up-trace``. With ``soundc`` the instrument
    sound speed is corrected to in-situ (``ss/sv``) exactly as getdpth does on sweep 1.
    """
    dt = centred_dt(d.time)
    wm = get_mean_w(d)

    if soundc:
        zzd0 = np.cumsum(wm * dt)
        zzu0 = np.flip(np.cumsum(np.flip(-wm * dt)))
        idmax = int(np.argmax(zzd0))
        iumax = int(np.argmax(zzu0))
        imax = round((idmax + iumax) / 2)
        zmax = (zzd0[imax] + zzu0[imax]) / 2.0
        zz = np.concatenate([
            zzd0[:imax] / zzd0[imax] * zmax,
            zzu0[imax:] / zzu0[imax] * zmax,
        ])
        pp = _press(np.abs(zz))
        ss = _sounds(pp, d.temp, 34.5)
        sc = ss / d.sv
        wm = wm * sc

    zzd = np.cumsum(wm * dt)
    zzu = np.flip(np.cumsum(np.flip(-wm * dt)))
    return {"down_max": float(np.max(zzd)), "up_max": float(np.max(zzu)), "wm": wm}


# --------------------------------------------------------------------------- #
# CTD depth on ADCP ping time (getdpth's ctddepth==1 input)
# --------------------------------------------------------------------------- #
def ctd_depth_on_ping(
    cast: SeabirdCast, time: np.ndarray, *, lag_s: float = 0.0
) -> np.ndarray:
    """CTD depth [m, **negative down**] interpolated onto ADCP ping times.

    Builds an absolute CTD clock from the ``.cnv`` start time + elapsed seconds, converts
    pressure to depth (``p2z``), and linearly interpolates onto ``time`` (datetime64). This
    is the ``d.z`` that ``getdpth`` uses when CTD depth is available (``ctddepth==1``): for
    the GO-SHIP golden it reproduces the profile max depth to <1 m. ``lag_s`` shifts the CTD
    clock for the (small) ADCP/CTD time offset; the full ``besttlag`` estimator is pending,
    and for RB1606 the residual offset is <0.01 s, so the default of 0 is faithful.
    """
    c = cast.ctd
    start = np.datetime64(cast.header["start_time"])
    ctd_t = start + ((c.time_elapsed_s + lag_s) * 1e9).astype("timedelta64[ns]")
    z = p2z(c.pressure, c.meta["lat_ref"])                 # positive down
    tc = ctd_t.astype("datetime64[ns]").astype(np.int64) / 1e9
    ta = time.astype("datetime64[ns]").astype(np.int64) / 1e9
    g = np.isfinite(z) & np.isfinite(tc)
    return -np.interp(ta, tc[g], z[g])                     # negative down (d.z)


def bin_depths(d: LADCPData, z: np.ndarray) -> np.ndarray:
    """Per-bin depth ``izm`` [m, negative down] ``[nbin, nens]`` (getdpth ``d.izm``).

    Each cell's depth = instrument depth ``z`` plus the bin offset in stack order
    ``[fliplr(zu), -zd]`` (up bins above the instrument, down bins below).
    """
    offset = np.concatenate([d.zu[::-1], -d.zd]) if d.zu.size else -d.zd
    return z[None, :] + offset[:, None]


# --------------------------------------------------------------------------- #
# Bottom detection + below-bottom removal (getdpth.m bottom-detection part)
# --------------------------------------------------------------------------- #
def detect_bottom_and_flag(
    d: LADCPData,
    z: np.ndarray,
    hbot: np.ndarray,
    *,
    wm: np.ndarray | None = None,
    near_bottom_m: float = 200.0,
    dzbelow_cells: tuple[float, float] = (2.0, -1.0),
) -> dict:
    """Bottom depth + below-bottom / surface flagging (``getdpth.m`` bottom-detection part).

    Reproduces the two-sweep bottom finder: restrict to ensembles within ``near_bottom_m``
    of the deepest point with a valid ``hbot``; sweep 1 takes the median of ``hbot - z`` and
    flags gross outliers, sweep 2 fits a quadratic of ``hbot - z`` vs ensemble (with the
    legacy outlier trim) and evaluates it at the deepest ensemble for ``zbottom``. Bins
    deeper than ``zbottom`` (minus a per-sweep ``dzbelow`` margin in cell-heights) and
    up-bins above the surface are flagged; the count of flagged cells with finite ``ru`` is
    the golden log's *values removed below bottom*.

    ``z`` is the depth time series [m, negative down] (CTD depth on ping time for the golden;
    see :func:`ctd_depth_on_ping`). ``hbot`` is the cleaned bottom distance [m]
    (see :func:`ladcp.ix.bottom.own_bottom_track`). Returns ``{zbottom, zbottomerror,
    ibottom, maxdepth, izmflag, n_removed}``; ``zbottom`` is NaN when no bottom is found.
    """
    if wm is None:
        wm = get_mean_w(d)
    z = np.asarray(z, float)
    hb = np.asarray(hbot, float).copy()             # mutated: outliers -> NaN between sweeps
    zd, zu, ru = d.zd, d.zu, d.ru

    maxdepth = float(np.max(-z))
    ibottom = int(np.argmax(-z))
    dzcell = float(np.nanmedian(np.abs(np.diff(zd))))
    dzbelow = [dzbelow_cells[0] * dzcell, dzbelow_cells[1] * dzcell]
    zz = -z                                          # positive depth

    # combined-bin depth offset: stack order [flipud(up); down] -> [fliplr(zu), -zd]
    offset = np.concatenate([zu[::-1], -zd]) if zu.size else -zd
    izm = z[None, :] + offset[:, None]               # [nbin, nens] bin depth (negative down)
    izmflag = np.zeros((d.n_bin, d.n_ens))

    zbottom = np.nan
    zbottomerror = np.nan
    with _quiet_nan():
        for n in (1, 2):
            iok = np.where((zz.max() - zz < near_bottom_m) & (hb > 0) & (np.abs(wm) > 0))[0]
            if iok.size:
                if n == 1:
                    zbottom = float(np.nanmedian(hb[iok] - z[iok]))
                    zerr = zbottom - (hb[iok] - z[iok])
                else:
                    c1 = np.polyfit(iok, hb[iok] - z[iok], 1)
                    e0 = np.polyval(c1, iok) - (hb[iok] - z[iok])
                    keep = (np.abs(e0) < 1.5 * np.nanstd(e0)) | (np.abs(e0) < 50.0)
                    iok = iok[keep]
                    c2 = np.polyfit(iok, hb[iok] - z[iok], 2)
                    zerr = np.polyval(c2, iok) - (hb[iok] - z[iok])
                    zbottom = float(np.polyval(c2, ibottom))
                zbottomerror = float(np.nanmedian(np.abs(zerr)))
                outl = (np.abs(zerr) > 2.0 * np.nanstd(zerr)) | (np.abs(zerr) > 100.0)
                hb[iok[outl]] = np.nan
            # reject implausible bottoms (shallower than the profile, or too uncertain)
            zb = zbottom
            if (zbottom - maxdepth < -20.0) or (zbottomerror > 20.0):
                zb = np.nan
            if np.isfinite(zb):
                izmflag[izm < -zb - dzbelow[n - 1]] = np.nan
            if zu.size:
                izmflag[izm > -zu[0]] = np.nan       # up-bins above the surface

    n_removed = int(np.count_nonzero(np.isnan(izmflag) & np.isfinite(ru)))
    if not (np.isfinite(zbottom) and not ((zbottom - maxdepth < -20.0) or
                                          (zbottomerror > 20.0))):
        zbottom = np.nan
    return {
        "zbottom": zbottom,
        "zbottomerror": zbottomerror,
        "ibottom": ibottom,
        "maxdepth": maxdepth,
        "izmflag": izmflag,
        "n_removed": n_removed,
    }
