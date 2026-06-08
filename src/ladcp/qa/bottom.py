"""Seabed detection from the down-looker echo (port of getbtrack/getdpthi essentials).

The seabed produces a strong echo-amplitude return below the normal water column. The
reported seabed depth (:func:`detect_bottom` -> :func:`_stack_seabed`) exploits the fact
that the seabed depth is *constant* across pings: raw echo amplitude is stacked along
constant-depth loci ``D = z_package + range`` and the seabed is the deepest trial depth that
reinforces well above the per-ping background. This is robust to the transmission-loss tail
of the volume-scatterer target strength (``targ``), which sits at constant *range* (not
depth) and smears out -- an earlier per-ping ``targ``-peak pick was biased 50-150 m deep on
weak/far returns. A cast that nearly touched bottom leaves the bed inside the inner range-gate
dead zone with no stackable echo; there we fall back to the deepest depth the package reached
(a tight lower bound) and flag the result. The per-ping bottom *distance* (:func:`bottom_distance`,
for the QA figures and echo count) is localized on raw amplitude (:func:`_seabed_bin`).

When a CTD cast is supplied distances are sound-speed corrected (:func:`soundspeed_scale`,
legacy getdpthi ``sc=ss./d.sv``); the firmware sizes its range bins with a fixed onboard
sound speed (here 1450 m/s), so the true distance scales by the in-situ/onboard ratio.

Validated against the CTD altimeter (seabed = depth + altimeter, where altimeter < 28 m) on
the MORIA stations that approached close enough: median |error| ~1.2 m, max ~12 m, no station
worse than 20 m (vs median 1 m but max 153 m and 7 stations > 20 m for the old ``targ``-peak
pick). MORIA-80 golden ``p.zbottom`` 1079.02; this gives ~1080.8 m (altimeter 1079.5 m).
See ``scripts/validate_seabed_altimeter.py`` for the scorer.
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
# Per-ping seabed distance (bottom_distance, for the QA figures + echo count) is localized on
# the *raw* echo amplitude -- not the volume-scatterer target strength targ(), whose range-gain
# over-amplifies far bins and pulls a per-ping pick metres deep on weak/far returns. The seabed
# is the deepest topographically-prominent peak, which also passes over mid-water scattering
# layers above it (see _seabed_bin).
_SEABED_MIN_PROM = 8.0   # dB, min peak prominence to call a per-ping return
# The reported seabed depth (detect_bottom -> _stack_seabed) stacks raw amplitude along
# constant-depth loci, which the targ tail can't bias because it sits at constant range.
_SEABED_STACK_DD = 1.0     # depth-grid step for the stack [m]
_SEABED_STACK_PROM = 5.0   # min prominence on the stack curve to be a candidate [dB]
_SEABED_STACK_MIN = 15.0   # min stacked amplitude over background to trust the echo [dB]
_SEABED_REFINE_WIN = 30.0  # refine zbottom from per-ping picks within this of the stack [m]
_BOTTOM_ERR_WARN = 10.0  # seabed scatter above this [m] -> flag the depth as uncertain
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


def _parabola_vertex(col: np.ndarray, k: int) -> float:
    """3-point parabola-refined fractional bin index of the peak at integer bin ``k``."""
    if k <= 0 or k >= col.size - 1:
        return float(k)
    y1, y2, y3 = col[k - 1], col[k], col[k + 1]
    denom = y1 - 2.0 * y2 + y3
    if not np.isfinite(denom) or denom == 0.0:
        return float(k)
    return k + 0.5 * (y1 - y3) / denom


def _seabed_bin(col: np.ndarray, gate: np.ndarray, *,
                min_prom: float = _SEABED_MIN_PROM) -> float:
    """Fractional range-bin of the seabed in one ping's amplitude profile, or NaN.

    The seabed is the *deepest topographically-prominent* peak (>= ``min_prom`` dB) of the
    raw echo amplitude within ``gate``. Prominence -- height above the valley separating it
    from any taller neighbour -- distinguishes a real reflector (the seabed, or a mid-water
    scattering layer) from the gradual reverberation/transmission-loss tail that has no dip
    before it. Taking the *deepest* prominent peak passes over scattering layers above the
    bed; using raw amplitude (not ``targ``) avoids the volume-scatterer range-gain that
    biases the pick deep.
    """
    from scipy.signal import find_peaks

    c = np.where(gate, col, np.nan)
    if np.isfinite(c).sum() < 5:
        return np.nan
    floor = float(np.nanpercentile(c, 25))
    cf = np.nan_to_num(c, nan=floor)
    peaks, _ = find_peaks(cf, prominence=min_prom)
    if peaks.size:
        return _parabola_vertex(cf, int(peaks[-1]))    # deepest prominent peak = seabed
    kmax = int(np.nanargmax(c))                         # fallback: a lone strong return
    if c[kmax] - floor < min_prom:
        return np.nan
    return _parabola_vertex(cf, kmax)


def bottom_distance(dh: DualHead) -> np.ndarray:
    """Per-ping distance to the seabed echo [m], NaN where no valid bottom echo.

    Localized on the raw echo amplitude (deepest strong return; see :func:`_seabed_bin`)
    rather than the target strength, then gated to the ``_BTRK_RANGE`` window.
    """
    d = dh.down
    zd = (d.meta.get("dist_first_m", d.blank_m + d.cell_m / 2.0)
          + np.arange(d.n_cells) * d.cell_m).astype(float)
    ea = np.nanmedian(d.echo, axis=0) * 0.45               # beams -> dB (counts*0.45)
    gate = (zd > _BTRK_RANGE[0]) & (zd < _BTRK_RANGE[1])
    n = ea.shape[1]
    hbot = np.full(n, np.nan)
    for j in range(n):
        kb = _seabed_bin(ea[:, j], gate)
        if np.isfinite(kb):
            hbot[j] = float(np.interp(kb, np.arange(zd.size), zd))
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
    zc = ctd_depth(ctd)                                        # +down, non-monotonic (up+down)
    order = np.argsort(zc)
    ss = np.interp(sync.z_on_ping, zc[order], css[order],
                   left=css[order][0], right=css[order][-1])
    sc = ss / sv
    sc[~np.isfinite(sc)] = 1.0
    return sc


def detect_bottom(dh: DualHead, sync: SyncResult,
                  ctd: CTDTimeSeries | None = None) -> BottomResult:
    """Estimate seabed depth by depth-stacking the down-looker echo.

    ``hbot`` (per-ping seabed distance, :func:`bottom_distance`) drives the QA figures and
    echo count; the reported ``zbottom`` comes from :func:`_stack_seabed`, which is robust
    to the transmission-loss tail that biased a per-ping pick deep. ``error`` is the scatter
    of the per-ping seabed depths about ``zbottom`` (NaN when no echo stacked, i.e. the
    fallback used the deepest package depth). When ``ctd`` is supplied distances are
    sound-speed corrected (:func:`soundspeed_scale`).
    """
    hbot_raw = bottom_distance(dh)
    sc = soundspeed_scale(dh, sync, ctd) if ctd is not None else np.ones_like(hbot_raw)
    hbot_raw = hbot_raw * sc
    z = sync.z_on_ping
    botdepth = z + hbot_raw                                # per-ping seabed depth (raw pick)

    d = dh.down
    zd = (d.meta.get("dist_first_m", d.blank_m + d.cell_m / 2.0)
          + np.arange(d.n_cells) * d.cell_m).astype(float)
    ea = np.nanmedian(d.echo, axis=0) * 0.45
    zstack, is_echo = _stack_seabed(z, ea, zd, sc)

    # the stack anchors the seabed region; the precise per-ping raw picks lying near it are
    # the pings that actually saw the bed -- their median refines zstack to ~metre precision
    # and excludes mid-water scatterers (and the deep tail). No echo stacked -> keep the
    # deepest-package-depth fallback and report no echoes.
    near = np.isfinite(botdepth) & (np.abs(botdepth - zstack) < _SEABED_REFINE_WIN)
    if is_echo and int(near.sum()) >= 5:
        zbottom = float(np.median(botdepth[near]))
        near = np.isfinite(botdepth) & (np.abs(botdepth - zbottom) < _SEABED_REFINE_WIN)
        error = float(np.median(np.abs(botdepth[near] - zbottom)))
    else:
        zbottom = zstack
        error = np.nan
    hbot = np.where(near, hbot_raw, np.nan)                # real bottom echoes (for the figure)
    n_valid = int(np.isfinite(hbot).sum()) if is_echo else 0
    return BottomResult(zbottom, error, hbot, n_valid)


def _stack_seabed(z: np.ndarray, ea: np.ndarray, zd: np.ndarray,
                  sc: np.ndarray) -> tuple[float, bool]:
    """Seabed depth by stacking raw echo amplitude along constant-depth loci.

    The seabed depth is constant across pings, so a true bottom return reinforces when the
    amplitude is summed along ``D = z_ping + range`` for trial depths ``D``; the
    transmission-loss / reverberation tail sits at constant *range* (not depth) and smears
    out. The seabed is the *deepest* trial depth whose stacked amplitude stands
    ``_SEABED_STACK_MIN`` dB above the per-ping background -- deeper than any mid-water
    scattering layer, and immune to the range-gain that pulls a per-ping ``targ`` pick deep.
    When nothing stacks that strongly there is no reliable echo (typically a cast that
    nearly touched bottom, so the bed sat in the inner range-gate dead zone): fall back to
    the deepest depth the package reached, a tight lower bound on the seabed.
    """
    from scipy.signal import find_peaks

    z = np.asarray(z, float)
    zmax = float(np.nanmax(z))
    lo, hi = _BTRK_RANGE
    bg = np.nanpercentile(ea, 25, axis=0)                  # per-ping background [dB]
    grid = np.arange(zmax - 20.0, zmax + hi, _SEABED_STACK_DD)
    stack = np.full(grid.size, np.nan)
    for i, dcand in enumerate(grid):
        r = (dcand - z) / sc                              # range each ping must see [m]
        ok = np.isfinite(r) & (r > lo) & (r < hi)
        if ok.sum() < 10:
            continue
        idx = np.where(ok)[0]
        amp = np.array([np.interp(r[j], zd, ea[:, j], left=np.nan, right=np.nan)
                        for j in idx])
        resid = amp - bg[idx]
        if np.isfinite(resid).any():
            stack[i] = np.nanmedian(resid)
    fin = np.isfinite(stack)
    if fin.sum() < 3:
        return zmax, False
    filled = np.nan_to_num(stack, nan=float(np.nanmin(stack[fin])))
    peaks, _ = find_peaks(filled, prominence=_SEABED_STACK_PROM)
    strong = [p for p in peaks if filled[p] >= _SEABED_STACK_MIN]
    if not strong:
        return zmax, False                                # no reliable echo -> deepest reached
    k = strong[-1]                                        # deepest strong stacked peak = seabed
    kfrac = _parabola_vertex(filled, k)
    return float(np.interp(kfrac, np.arange(grid.size), grid)), True


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
    ok = np.isfinite(b.zbottom) and b.error <= _BOTTOM_ERR_WARN
    return Metric(
        "bottom_depth", round(b.zbottom, 1), "m",
        Status.OK if ok else Status.WARN,
        threshold=_BOTTOM_ERR_WARN,
        source_stage="qa.bottom",
        note=f"+/- {b.error:.2f} m from {b.n_valid} bottom echoes",
    )


@dataclass
class BtrkDiagnostics:
    """Own vs RDI bottom-track comparison (port of ``checkbtrk`` essentials, Figure 13).

    Both estimates are the earth-frame package velocity over ground; the RDI firmware
    bottom track and our own near-seabed-cell track are independent, so their agreement
    is the headline bottom-track quality check. Velocities are stored as finite-only 1-D
    arrays for plotting; ``*_bias`` is ``median(own) - median(rdi)`` per component (NaN if
    no RDI track), ``*_std`` the own-track scatter, ``roughness`` the std of the detected
    bottom distance.
    """

    own_u: np.ndarray
    own_v: np.ndarray
    own_w: np.ndarray
    rdi_u: np.ndarray
    rdi_v: np.ndarray
    rdi_w: np.ndarray
    hbot: np.ndarray
    u_bias: float
    v_bias: float
    w_bias: float
    u_std: float
    v_std: float
    w_std: float
    roughness: float
    n_own: int
    n_rdi: int


def btrk_diagnostics(bt: BottomTrack, dh) -> BtrkDiagnostics:
    """Compare our own per-ping bottom track to the RDI firmware track (both earth frame).

    Our ``bt.bvel`` is ``-u_package``; RDI reports the bottom's velocity relative to the
    transducer, also ``-u_package``, so the two share a sign and are compared directly.
    Returns the finite velocity samples plus inter-method bias/scatter and roughness.
    """
    own = bt.bvel[np.isfinite(bt.bvel)]
    own_u, own_v = np.real(own), np.imag(own)
    own_w = bt.bw[np.isfinite(bt.bw)]

    rdi_u = rdi_v = rdi_w = np.array([])
    btv = getattr(dh.down, "bt_vel", None)
    if btv is not None:
        rdi_u = btv[0][np.isfinite(btv[0])]
        rdi_v = btv[1][np.isfinite(btv[1])]
        rdi_w = btv[2][np.isfinite(btv[2])]

    def bias(o, r):
        return float(np.median(o) - np.median(r)) if o.size and r.size else np.nan

    return BtrkDiagnostics(
        own_u=own_u, own_v=own_v, own_w=own_w,
        rdi_u=rdi_u, rdi_v=rdi_v, rdi_w=rdi_w,
        hbot=bt.hbot[np.isfinite(bt.hbot)],
        u_bias=bias(own_u, rdi_u), v_bias=bias(own_v, rdi_v), w_bias=bias(own_w, rdi_w),
        u_std=float(np.std(own_u)) if own_u.size else np.nan,
        v_std=float(np.std(own_v)) if own_v.size else np.nan,
        w_std=float(np.std(own_w)) if own_w.size else np.nan,
        roughness=float(np.std(bt.hbot[np.isfinite(bt.hbot)])) if bt.n_valid else np.nan,
        n_own=int(own_u.size), n_rdi=int(rdi_u.size),
    )
