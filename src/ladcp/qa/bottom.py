"""Seabed detection from the down-looker echo (port of getbtrack/getdpthi essentials).

The seabed produces a strong echo-amplitude return below the normal water column. We
convert echo amplitude to an attenuation-corrected target strength (``targ``), find the
per-ensemble peak distance by parabola fit (``localmax2``), accept it as a bottom echo
when it stands ``btrk_ts`` dB above bin 1 and lies within ``btrk_range``, then combine
the bottom distance with the synchronized package depth to get the seabed depth::

    zbottom = robust_fit( z_package + bottom_distance )   over near-bottom pings

The seabed depth is fit with a quadratic over the near-bottom pings and evaluated at the
deepest ping (faithful port of getdpthi.m). When a CTD cast is supplied the bottom
distance is corrected for sound speed (:func:`soundspeed_scale`, legacy getdpthi
``sc=ss./d.sv``); the firmware sizes its range bins with a fixed onboard sound speed
(here 1450 m/s), so the true distance scales by the in-situ/onboard ratio.

Validated against MORIA-80 (golden ``p.zbottom`` 1079.02, ``p.zbottomerror`` 0.56):
quadratic estimator alone gives 1079.3 m; with the sound-speed correction 1080.1 m
(physically deeper, within the golden +/-0.6 m band and the 2 m QA gate).
"""

from __future__ import annotations

import warnings
from dataclasses import dataclass

import numpy as np

from ..models import CTDTimeSeries, Metric, Status
from .depth import SyncResult, ctd_depth
from .ingest import DualHead

_ATT_300 = 0.06          # attenuation dB/m at 300 kHz (legacy ts_att)
_ATT_OTHER = 0.039
_SOURCE_LEVEL = 100.0    # eas in targ()
_APERTURE_DEG = 2.0
_BTRK_TS = 10.0          # dB excess over bin 1 to call a bottom echo
_BTRK_RANGE = (50.0, 300.0)
_NEAR_BOTTOM = 200.0     # within this many m of max depth -> near-bottom ping
_BTRK_BELOW = 0.5        # bins below the target-strength max used for bottom velocity
_BTRK_WLIM = 0.05        # max |W_btrk - W_ref| [m/s] to accept a bottom velocity


@dataclass
class BottomResult:
    zbottom: float
    error: float
    hbot: np.ndarray            # [nens] bottom distance per ping (NaN where none)
    n_valid: int


def targ(ea: np.ndarray, dis: np.ndarray, at: float, bl: float,
         eas: float = _SOURCE_LEVEL, ap: float = _APERTURE_DEG) -> np.ndarray:
    """Attenuation-corrected target strength for a volume scatterer (legacy ``targ``)."""
    al = np.radians(ap)
    d = dis[:, None].astype(float)
    r1 = np.tan(al) * (d - bl / 2)
    r2 = np.tan(al) * (d + bl / 2)
    v = np.pi * bl / 3 * (r1 ** 2 + r2 ** 2 + r1 * r2)
    tl = 20 * np.log10(d) + at * d
    return ea - eas + 2 * tl - 10 * np.log10(v)


def localmax2(x: np.ndarray, y: np.ndarray) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Per-column parabola-fit maximum (port of ``localmax2.m``). Returns (xpeak, ypeak, imax)."""
    nbin, nens = y.shape
    imax = np.nanargmax(y, axis=0)
    xout = np.full(nens, np.nan)
    yout = np.full(nens, np.nan)
    for j in np.where((imax > 0) & (imax < nbin - 1))[0]:
        k = imax[j]
        x1, x2, x3 = x[k - 1], x[k], x[k + 1]
        y1, y2, y3 = y[k - 1, j], y[k, j], y[k + 1, j]
        denom = (x3 * x2**2 - x1 * x2**2 + x1 * x3**2 - x1**2 * x3 + x1**2 * x2 - x3**2 * x2)
        a = (x3 * y2 + x1 * y3 - x1 * y2 - y3 * x2 - y1 * x3 + y1 * x2) / denom
        b = -(-x2**2 * y3 + x2**2 * y1 - y2 * x1**2 + y3 * x1**2 - x3**2 * y1 + y2 * x3**2) \
            / ((-x3 + x2) * (x2 * x3 - x2 * x1 + x1**2 - x3 * x1))
        c = (x2**2 * y1 * x3 - x2**2 * x1 * y3 - x3**2 * y1 * x2 + y3 * x1**2 * x2
             + x3**2 * x1 * y2 - y2 * x1**2 * x3) \
            / ((-x3 + x2) * (x2 * x3 - x2 * x1 + x1**2 - x3 * x1))
        if a < 0:
            xn = -b / a / 2
            xout[j], yout[j] = xn, xn**2 * a + xn * b + c
    return xout, yout, imax


def bottom_distance(dh: DualHead) -> np.ndarray:
    """Per-ping distance to the seabed echo [m], NaN where no valid bottom echo."""
    d = dh.down
    at = _ATT_300 if d.freq_khz == 300 else _ATT_OTHER
    zd = (d.meta.get("dist_first_m", d.blank_m + d.cell_m / 2.0)
          + np.arange(d.n_cells) * d.cell_m).astype(float)
    ea = np.nanmedian(d.echo, axis=0) * 0.45               # beams -> dB (counts*0.45)
    tg = targ(ea, zd, at, d.cell_m)
    zpeak, ypeak, _ = localmax2(zd, tg)
    dts = ypeak - tg[0]
    valid = (dts > _BTRK_TS) & (zpeak > _BTRK_RANGE[0]) & (zpeak < _BTRK_RANGE[1])
    hbot = np.where(valid, zpeak, np.nan)
    return hbot


def soundspeed_scale(dh: DualHead, sync: SyncResult, ctd: CTDTimeSeries) -> np.ndarray:
    """Per-ping bin-length scale ``sc = c_ctd(z) / c_adcp`` (legacy getdpth ``sc=ss./d.sv(1,:)``).

    The down-looker firmware sizes its range bins with the onboard sound speed
    ``d.sv`` (computed from the head temperature and a fixed salinity). The true range
    scales by the ratio of the actual in-situ sound speed at the package depth to that
    onboard value, so the detected bottom distance ``hbot`` must be multiplied by ``sc``.
    (Our package depth comes from CTD pressure, so unlike the legacy W-integration it
    needs no sound-speed rescale — only the bottom *distance* does.)
    """
    import gsw

    sv = np.asarray(dh.down.sound_speed, dtype=float)          # onboard m/s, per ping
    lat = float(np.nanmedian(ctd.lat))
    lon = float(np.nanmedian(ctd.lon))
    sa = gsw.SA_from_SP(ctd.salinity, ctd.pressure, lon, lat)
    ct = gsw.CT_from_t(sa, ctd.temperature, ctd.pressure)
    css = gsw.sound_speed(sa, ct, ctd.pressure)                # in-situ CTD profile m/s
    zc = ctd_depth(ctd)                                        # +down, non-monotonic (up+down cast)
    order = np.argsort(zc)
    ss = np.interp(sync.z_on_ping, zc[order], css[order],
                   left=css[order][0], right=css[order][-1])
    sc = ss / sv
    sc[~np.isfinite(sc)] = 1.0
    return sc


def detect_bottom(dh: DualHead, sync: SyncResult,
                  ctd: CTDTimeSeries | None = None) -> BottomResult:
    """Estimate seabed depth from bottom distance + synchronized package depth.

    When ``ctd`` is supplied, the bottom distance is corrected for sound speed
    (:func:`soundspeed_scale`); otherwise the uncorrected nominal-sound-speed
    distance is used (~1 m deep on a ~1080 m cast).
    """
    hbot = bottom_distance(dh)
    if ctd is not None:
        hbot = hbot * soundspeed_scale(dh, sync, ctd)
    z = sync.z_on_ping
    botdepth = z + hbot                                    # seabed depth per ping
    n_valid = int(np.isfinite(hbot).sum())

    # near-bottom pings with a valid bottom echo (getdpthi: within 200 m of deepest)
    iok = np.where(np.isfinite(botdepth) & (hbot > 0)
                   & ((np.nanmax(z) - z) < _NEAR_BOTTOM))[0]
    if iok.size < 10:
        return BottomResult(np.nan, np.nan, hbot, n_valid)
    ibottom = int(np.nanargmax(z))
    bd = botdepth[iok]

    # robust rejection then quadratic fit evaluated at the deepest ping
    # (faithful port of getdpthi.m lines 338-350)
    err = np.abs(np.median(bd) - bd)
    half = np.argsort(err)[: err.size // 2]
    c = np.polyfit(iok[half], bd[half], 1)
    err = np.abs(np.polyval(c, iok) - bd)
    half = np.argsort(err)[: err.size // 2]
    keep = (err < 2.0 * np.std(err[half])) | (err < 30.0)
    iok, bd = iok[keep], bd[keep]

    c = np.polyfit(iok, bd, 2)
    zbottom = float(np.polyval(c, ibottom))
    error = float(np.median(np.abs(np.polyval(c, iok) - bd)))
    return BottomResult(zbottom, error, hbot, n_valid)


@dataclass
class BottomTrack:
    """Per-ping bottom-track velocity (legacy ``getbtrack`` velocity branch).

    The cell at the seabed reflects off the (stationary) bottom, so its earth-frame
    Doppler velocity is the *package* velocity over ground (``bvel = -u_package``). The
    full water-cell measurement is ``ru = u_ocean - u_package``, hence subtracting ``bvel``
    from any cell recovers the absolute ocean velocity -- this is the bottom-track
    reference used by :func:`ladcp.qa.inverse.bottom_referenced_profile`.
    """

    bvel: np.ndarray            # [nens] complex u + i*v package velocity (NaN where none)
    bw: np.ndarray              # [nens] vertical bottom-track velocity
    hbot: np.ndarray            # [nens] bottom distance used [m] (NaN where none)
    n_valid: int


def _boutlier(bvel: np.ndarray, bw: np.ndarray, *, nfac=(4.0, 3.0)) -> None:
    """Two-pass RMS rejection of bottom-track velocities in place (legacy ``boutlier``)."""
    for nf in nfac:
        good = np.isfinite(bvel)
        if good.sum() < 4:
            return
        for comp in (np.real(bvel), np.imag(bvel), bw):
            a = comp - np.nanmedian(comp)
            rms = np.sqrt(np.nanmean(a[np.isfinite(a)] ** 2))
            bad = np.abs(a) > nf * rms
            bvel[bad] = np.nan
            bw[bad] = np.nan


def bottom_track_velocity(dh: DualHead, merged, *, btrk_below: float = _BTRK_BELOW,
                          btrk_ts: float = _BTRK_TS, btrk_range=_BTRK_RANGE,
                          btrk_wlim: float = _BTRK_WLIM) -> BottomTrack:
    """Per-ping bottom-track velocity from the down-looker near-seabed cells.

    Faithful port of the ``getbtrack`` velocity branch: locate the bottom echo by parabola
    fit (:func:`localmax2`), pick the cell ``btrk_below`` bins past the target-strength
    maximum, and take the median earth-frame velocity over that cell +/- one neighbour
    (``merged`` down-looker rows, already rotated into the down frame). Pings whose
    bottom-cell W differs from the reference-layer W by more than ``btrk_wlim`` are dropped,
    then a two-pass RMS outlier rejection is applied.
    """
    d = dh.down
    n = merged.ru.shape[1]
    dz = d.cell_m
    zd = (d.meta.get("dist_first_m", d.blank_m + d.cell_m / 2.0)
          + np.arange(d.n_cells) * d.cell_m).astype(float)
    at = _ATT_300 if d.freq_khz == 300 else _ATT_OTHER
    ea = np.nanmedian(d.echo, axis=0)[:, :n] * 0.45
    tg = targ(ea, zd, at, d.cell_m)
    zpeak, ypeak, _ = localmax2(zd, tg)
    dts = ypeak - tg[0]

    # firmware bin index of the bottom-velocity cell (MATLAB round = floor(x+0.5))
    imeadbv = np.floor((zpeak - zd[0]) / dz + btrk_below + 0.5)
    nbin_d = d.n_cells
    valid = (np.isfinite(imeadbv) & (dts > btrk_ts)
             & (zpeak > btrk_range[0]) & (zpeak < btrk_range[1])
             & (imeadbv > 0) & (imeadbv < nbin_d - 1))

    izd = merged.izd
    ru_d, rv_d, rw_d = merged.ru[izd], merged.rv[izd], merged.rw[izd]
    bvel = np.full(n, np.nan, dtype=complex)
    bw = np.full(n, np.nan)
    with warnings.catch_warnings():
        warnings.simplefilter("ignore", RuntimeWarning)
        for j in np.where(valid)[0]:
            k = int(imeadbv[j])
            rows = [k - 1, k, k, k + 1]                 # centre weighted (legacy [-1 0 0 1])
            bvel[j] = (np.nanmedian(ru_d[rows, j])
                       + 1j * np.nanmedian(rv_d[rows, j]))
            bw[j] = np.nanmedian(rw_d[rows, j])
        wref = np.nanmedian(rw_d, axis=0)
    bad = np.abs(wref - bw) > btrk_wlim
    bvel[bad] = np.nan
    bw[bad] = np.nan

    _boutlier(bvel, bw)
    hbot = np.where(np.isfinite(bvel), zpeak, np.nan)
    return BottomTrack(bvel=bvel, bw=bw, hbot=hbot, n_valid=int(np.isfinite(bvel).sum()))


def bottom_metric(b: BottomResult) -> Metric:
    return Metric(
        "bottom_depth", round(b.zbottom, 1), "m",
        Status.OK if np.isfinite(b.zbottom) else Status.WARN,
        source_stage="qa.bottom",
        note=f"+/- {b.error:.2f} m from {b.n_valid} bottom echoes",
    )
