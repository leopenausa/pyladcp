"""Seabed detection from the down-looker echo (port of getbtrack/getdpthi essentials).

The seabed produces a strong echo-amplitude return below the normal water column. The
reported seabed depth (:func:`detect_bottom` -> :func:`_stack_seabed`) exploits the fact
that the seabed depth is *constant* across pings: raw echo amplitude is stacked along
constant-depth loci ``D = z_package + range`` and the seabed is the deepest trial depth that
reinforces well above the per-ping background. This is robust to the transmission-loss tail
of the volume-scatterer target strength (``targ``), which sits at constant *range* (not
depth) and smears out. A cast that nearly touched bottom leaves the bed inside the inner
range-gate dead zone with no stackable echo (only weak or multiple returns); there the
deepest depth the package reached is the seabed to within a few metres and is reported as a
flagged lower bound (``is_fallback``), or NaN when no near-bottom echo confirms the touch.
The per-ping bottom *distance* (:func:`bottom_distance`, for the QA figures, the stack
refinement and the echo count) is the single strongest ``targ`` return per ping (legacy
``getbtrack``/``localmax2``): tight and flat at the constant-depth bed, and -- unlike a
"deepest prominent peak" -- it does not wander onto the deeper multiples / previous-ping
interference whose constant *range* makes ``z + hbot`` cross the seabed line.

When a CTD cast is supplied distances are sound-speed corrected (:func:`soundspeed_scale`,
legacy getdpthi ``sc=ss./d.sv``); the firmware sizes its range bins with a fixed onboard
sound speed (here 1450 m/s), so the true distance scales by the in-situ/onboard ratio.

Validated against the CTD altimeter (seabed = depth + altimeter, where altimeter < 28 m) and
the operator echo-sounder. The 28/40 MORIA stations that lock the stack: median |error| ~1.3 m,
max ~12 m. The 12 near-touch fallback casts: ``zmax`` is within 2-11 m of the altimeter /
echo-sounder on every one. MORIA-80 golden ``p.zbottom`` 1079.02; this gives ~1080.8 m
(altimeter 1079.5 m). See ``scripts/validate_seabed_altimeter.py`` for the scorer.
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
# Per-ping seabed distance (bottom_distance, for the QA figures, the refinement and the echo
# count) is the single strongest target-strength return per ping (targ + localmax2, legacy
# getbtrack). This is tight and flat at the constant-depth seabed on casts with a clean return;
# a "deepest prominent raw peak" detector instead wanders onto multiples / previous-ping
# interference (constant *range*, so z+hbot tracks the package and crosses the seabed line).
# targ's volume-scatterer range-gain can pull this pick deep on *weak/far* returns, but that no
# longer reaches the reported zbottom, which comes from the depth-stack (or the zmax near-touch
# fallback), not from pooling these per-ping picks.
# The reported seabed depth (detect_bottom -> _stack_seabed) stacks raw amplitude along
# constant-depth loci, which the targ tail can't bias because it sits at constant range.
_SEABED_STACK_DD = 1.0     # depth-grid step for the stack [m]
_SEABED_STACK_PROM = 5.0   # min prominence on the stack curve to be a candidate [dB]
_SEABED_STACK_MIN = 15.0   # min stacked amplitude over background to trust the echo [dB]
_MULTIPLE_TOL = 0.2        # |depth ratio - 2 or 3| within this -> deeper peak is a bottom multiple
_SEABED_REFINE_WIN = 30.0  # refine zbottom from per-ping picks within this of the stack [m]
_NEAR_BOTTOM = 200.0       # near-bottom window (deepest-package-depth minus this) [m] (getdpthi)
_NEARTOUCH_MIN = 3         # min near-bottom bottom echoes to trust a zmax near-touch fallback
_BOTTOM_ERR_WARN = 10.0  # seabed scatter above this [m] -> flag the depth as uncertain
_SEABED_XCHECK = 25.0    # |stack - legacy-polyfit| above this [m] -> flag disagreement (WARN)
_STACK_COVER_MIN = 0.4   # min package-depth support span (fraction of the geometrically visible
                         # range) for a stack lock to be a real bed and not a constant-range
                         # artifact -- see _support_cover / detect_bottom
_BTRK_BELOW = 0.5        # bins below the target-strength max used for bottom velocity
_BTRK_WLIM = 0.05        # max |W_btrk - W_ref| [m/s] to accept a bottom velocity


@dataclass
class BottomResult:
    zbottom: float
    error: float
    hbot: np.ndarray            # [nens] bottom distance per ping (NaN where none)
    n_valid: int
    is_fallback: bool = False   # True when the stack failed and zbottom is the zmax near-touch
                                # lower bound (or NaN) rather than a stacked echo lock
    zbottom_legacy: float = np.nan   # faithful getdpthi.m polyfit seabed -- DIAGNOSTIC cross-check
                                     # only, never the reported value (see _legacy_seabed)
    legacy_error: float = np.nan     # legacy self-reported zbottomerror [m]


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
    """Per-ping distance to the seabed echo [m], NaN where no valid bottom echo.

    The single strongest target-strength return per ping, parabola-refined
    (:func:`localmax2` on :func:`targ`), gated to ``_BTRK_RANGE`` and required to stand
    ``_BTRK_TS`` dB above bin 1 -- the legacy ``getbtrack`` bottom distance. Picking the
    *strongest* return (rather than the deepest prominent peak) keeps the per-ping seabed
    flat at constant depth and off the deeper multiples / previous-ping interference, whose
    constant *range* makes ``z + hbot`` track the package and cross the seabed line.
    """
    d = dh.down
    at = _ATT_300 if d.freq_khz == 300 else _ATT_OTHER
    zd = (d.meta.get("dist_first_m", d.blank_m + d.cell_m / 2.0)
          + np.arange(d.n_cells) * d.cell_m).astype(float)
    ea = np.nanmedian(d.echo, axis=0) * 0.45               # beams -> dB (counts*0.45)
    tg = targ(ea, zd, at, d.cell_m)
    zpeak, ypeak, _ = localmax2(zd, tg)
    dts = ypeak - tg[0]
    valid = (dts > _BTRK_TS) & (zpeak > _BTRK_RANGE[0]) & (zpeak < _BTRK_RANGE[1])
    return np.where(valid, zpeak, np.nan)


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
    """Estimate seabed depth: a depth-stack echo lock, else a near-touch ``zmax`` fallback.

    The reported ``zbottom`` is the depth where bottom echoes reinforce across pings
    (:func:`_stack_seabed`), refined to ~metre precision by the per-ping seabed picks
    (:func:`bottom_distance`) lying near it. When nothing stacks -- typically a cast that
    nearly touched bottom, so the bed sat in the inner range-gate dead zone and the only
    returns are weak or deeper multiples -- the package's deepest depth ``zmax`` is the
    seabed to within a few metres (validated <=11 m vs the CTD altimeter and the operator
    echo-sounder on every MORIA fallback cast), and is reported as a flagged lower bound
    *provided* at least ``_NEARTOUCH_MIN`` near-bottom echoes confirm the package reached the
    bed. With no such echoes the bottom is genuinely unknown and ``zbottom`` is NaN (legacy
    "no bottom found" -> no below-bottom / side-lobe masking downstream, gate skipped).
    ``is_fallback`` marks both fallback cases; ``error`` is always a real scatter, never NaN
    on a finite ``zbottom``. When ``ctd`` is supplied distances are sound-speed corrected
    (:func:`soundspeed_scale`).
    """
    hbot_raw = bottom_distance(dh)
    sc = soundspeed_scale(dh, sync, ctd) if ctd is not None else np.ones_like(hbot_raw)
    hbot_raw = hbot_raw * sc
    z = sync.z_on_ping
    botdepth = z + hbot_raw                                # per-ping seabed depth
    zmax = float(np.nanmax(z))
    # faithful legacy getdpthi polyfit, computed as a cross-check only (never reported)
    zb_leg, leg_err = _legacy_seabed(z, hbot_raw, zmax)

    d = dh.down
    zd = (d.meta.get("dist_first_m", d.blank_m + d.cell_m / 2.0)
          + np.arange(d.n_cells) * d.cell_m).astype(float)
    ea = np.nanmedian(d.echo, axis=0) * 0.45
    zstack, is_echo = _stack_seabed(z, ea, zd, sc)

    if is_echo:
        # support-spread guard: a real bed at a constant depth is seen by pings spanning a wide
        # range of package depths (range = zbottom - z stays in the echo band as the package
        # descends), so the per-ping seabed picks near the lock cover most of the geometrically
        # visible span. A constant-*range* artifact (multiple / previous-ping interference) only
        # crosses the constant-depth lock line over a narrow band of package depths, so its
        # support is a thin cluster. Reject a narrow-support lock and fall back to the near-touch
        # zmax bound rather than report the artifact (FDCCC1_001: 242 m lock, golden 127 m;
        # support covers only 26% vs >=61% on every genuine lock across 40 MORIA + FDCCC casts).
        near0 = np.isfinite(botdepth) & (np.abs(botdepth - zstack) < _SEABED_REFINE_WIN)
        is_echo = _support_cover(z[near0], zstack, zmax) >= _STACK_COVER_MIN

    if is_echo:
        # the parabola-refined stack peak is the reported seabed (unbiased -- the volume-
        # scatterer range-gain that pulls a per-ping targ pick ~1-2 m deep does not enter it).
        # The per-ping picks lying near it -- the pings that actually saw the bed -- give the
        # scatter (error), the echo count and the figure dots.
        zbottom = zstack
        near = np.isfinite(botdepth) & (np.abs(botdepth - zbottom) < _SEABED_REFINE_WIN)
        error = float(np.median(np.abs(botdepth[near] - zbottom))) if near.any() else 0.0
        hbot = np.where(near, hbot_raw, np.nan)
        return BottomResult(zbottom, error, hbot, int(np.isfinite(hbot).sum()),
                            zbottom_legacy=zb_leg, legacy_error=leg_err)

    # stack failed (weak or dead-zone bottom echo). The package nearly touched, so the deepest
    # depth it reached is the seabed to within a few metres (validated <=11 m vs altimeter +
    # echo-sounder) -- a flagged lower bound. Deriving a depth from the per-ping echoes is worse:
    # on these casts the bed sits in the inner range-gate dead zone and the only returns are
    # constant-range multiples (which a cluster fit happily locks onto, 10-30 m off). Only echoes
    # at or below zmax can be the bed (one mapping *above* zmax is mid-water); restrict the figure
    # picks to those so they sit below the lower-bound line instead of crossing it.
    near = np.isfinite(botdepth) & (z > zmax - _NEAR_BOTTOM)
    if int(near.sum()) < _NEARTOUCH_MIN:                   # no near-bottom echo -> bed unknown
        return BottomResult(np.nan, np.nan, np.full_like(hbot_raw, np.nan), 0, is_fallback=True,
                            zbottom_legacy=zb_leg, legacy_error=leg_err)
    bed = near & (botdepth >= zmax) & (botdepth < zmax + _SEABED_REFINE_WIN)
    error = (float(np.median(botdepth[bed] - zmax)) if int(bed.sum()) >= 3
             else float(_BOTTOM_ERR_WARN))                 # honest uncertainty, never NaN
    hbot = np.where(bed, hbot_raw, np.nan)
    return BottomResult(zmax, error, hbot, int(np.isfinite(hbot).sum()), is_fallback=True,
                        zbottom_legacy=zb_leg, legacy_error=leg_err)


def _support_cover(z_support: np.ndarray, zbottom: float, zmax: float) -> float:
    """Fraction of the geometrically visible package-depth span that the lock's echoes cover.

    A real bed at ``zbottom`` is visible whenever the package range ``zbottom - z`` falls in the
    bottom-echo band ``_BTRK_RANGE``, i.e. for package depths ``z`` in
    ``[zbottom - hi, min(zmax, zbottom - lo)]`` -- a span the descent actually traverses. The
    per-ping seabed picks near a genuine lock spread across most of that span; a constant-range
    artifact masquerading as a constant-depth bed only does so over a thin band of package
    depths. Returns ``support_span / visible_span`` (1.0 when the geometry leaves no span to
    cover, so the guard never rejects on a degenerate denominator).
    """
    z_support = np.asarray(z_support, float)
    if z_support.size < 2:
        return 0.0
    lo, hi = _BTRK_RANGE
    visible = min(float(zmax), zbottom - lo) - max(0.0, zbottom - hi)
    if visible <= 0.0:
        return 1.0
    return float(z_support.max() - z_support.min()) / visible


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
    # Vectorized stack: zd (bin depths) is shared across pings, so the trial-depth x ping
    # double loop over np.interp becomes pure array ops. Bit-identical to the scalar loop;
    # ~2-3x faster, and this is 60-70% of a velocity solve (see ladcp-velocity-perf note).
    nbin = zd.size
    R = (grid[:, None] - z[None, :]) / sc[None, :]        # [G, P] range each (depth,ping) needs
    ok = np.isfinite(R) & (R > lo) & (R < hi)             # [G, P] per-ping range gate
    idx = np.clip(np.searchsorted(zd, R.ravel(), side="right").reshape(R.shape) - 1,
                  0, nbin - 2)                            # lower zd-bin per (depth,ping)
    z0, z1 = zd[idx], zd[idx + 1]
    a0 = np.take_along_axis(ea, idx, axis=0)
    a1 = np.take_along_axis(ea, idx + 1, axis=0)
    with np.errstate(invalid="ignore", divide="ignore"):
        amp = a0 + (R - z0) / (z1 - z0) * (a1 - a0)       # linear interp of ea(zd) at R
    amp[~ok | (R < zd[0]) | (R > zd[-1])] = np.nan        # match np.interp left/right=nan + gate
    resid = amp - bg[None, :]                             # [G, P]
    with warnings.catch_warnings():
        warnings.simplefilter("ignore", RuntimeWarning)   # all-NaN rows -> NaN (gated below)
        stack = np.where((ok.sum(axis=1) >= 10) & np.isfinite(resid).any(axis=1),
                         np.nanmedian(resid, axis=1), np.nan)
    fin = np.isfinite(stack)
    if fin.sum() < 3:
        return zmax, False
    filled = np.nan_to_num(stack, nan=float(np.nanmin(stack[fin])))
    peaks, _ = find_peaks(filled, prominence=_SEABED_STACK_PROM)
    strong = [p for p in peaks if filled[p] >= _SEABED_STACK_MIN]
    if not strong:
        return zmax, False                                # no reliable echo -> deepest reached
    k = _pick_seabed_peak(grid, filled, strong)
    return float(grid[k]), True                           # 1 m grid; parabola refine overshoots


def _pick_seabed_peak(grid: np.ndarray, filled: np.ndarray, strong: list[int]) -> int:
    """Pick the seabed among strong stacked peaks (returns the chosen peak index).

    The seabed is normally the *deepest* strong peak -- deeper than any mid-water scattering
    layer. But a hard bed re-reflects surface<->bed, leaving weaker MULTIPLES at ~integer
    multiples of the bed depth; on a shallow cast these reinforce into their own strong peak
    (MORIA-89: true bed 86 m amp 46, multiple 170 m amp 24; echo-sounder 83 m). Demote the
    deepest peak to a shallower, *at-least-as-strong* strong peak when it sits at ~2x or ~3x
    that peak's depth -- a harmonic multiple, not a deeper bed. Single-peak casts (the other 39
    MORIA stations) never trigger this and are unchanged.
    """
    k = strong[-1]
    changed = True
    while changed:
        changed = False
        for j in strong:
            if grid[j] >= grid[k]:
                break                                     # only consider shallower strong peaks
            ratio = grid[k] / grid[j]
            if filled[j] >= filled[k] and (abs(ratio - 2.0) <= _MULTIPLE_TOL
                                           or abs(ratio - 3.0) <= _MULTIPLE_TOL):
                k = j                                     # deepest peak is a harmonic multiple
                changed = True
                break
    return k


def _legacy_seabed(z: np.ndarray, hbot: np.ndarray, maxdepth: float) -> tuple[float, float]:
    """Faithful port of the legacy ``getdpthi.m`` seabed polyfit (lines 327-424, n==n2 branch).

    DIAGNOSTIC cross-check only -- never the reported seabed. Legacy derives the bed purely from
    the per-ping bottom distances ``hbot`` in the near-bottom window: a robust median -> half-trim
    deg-1 -> deg-2 fit of ``hbot + depth`` vs ensemble, evaluated at the deepest ping, with a
    ``NaN`` gate when the self-error exceeds 20 m or the bed would sit implausibly above the
    deepest package depth. It is excellent on clean casts but locks *silently* onto constant-range
    multiples (its >20 m gate never fires there -- their scatter is ~0.3 m), which is exactly why
    :func:`detect_bottom` reports the constant-depth stack instead and only compares against this
    value to flag disagreement. Returns ``(zbottom, zbottomerror)``; ``(nan, nan)`` if unknown.
    """
    dz_neg = -np.asarray(z, dtype=float)                  # legacy d.z is negative-down
    hb = np.asarray(hbot, dtype=float)
    if not np.isfinite(dz_neg).any():
        return np.nan, np.nan
    ibottom = int(np.nanargmax(-dz_neg))                  # deepest ping (max(-d.z))
    maxnegz = float(np.nanmax(-dz_neg))
    iok = np.where(((maxnegz + dz_neg) < _NEAR_BOTTOM) & (hb > 0) & np.isfinite(hb))[0]
    if iok.size <= 2:
        return np.nan, np.nan
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")                   # polyfit conditioning on raw indices
        bd = hb[iok] - dz_neg[iok]                         # seabed depth per ping = hbot + depth
        resid = np.median(bd) - bd
        half = np.argsort(np.abs(resid))[: resid.size // 2]
        c1 = np.polyfit(iok[half], bd[half], 1)
        resid = np.polyval(c1, iok) - bd
        half = np.argsort(np.abs(resid))[: resid.size // 2]
        std_half = np.std(resid[half])
        keep = (np.abs(resid) < 2 * std_half) | (np.abs(resid) < 30)
        iok2 = iok[keep]
        if iok2.size < 3:
            return np.nan, np.nan
        bd2 = hb[iok2] - dz_neg[iok2]
        c2 = np.polyfit(iok2, bd2, 2)
        zbottom = float(np.polyval(c2, ibottom))
        err = float(np.median(np.abs(np.polyval(c2, iok2) - bd2)))
    if (zbottom - maxdepth < -(maxdepth * 0.01 + 10)) or (err > 20):
        return np.nan, err
    return zbottom, err


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
    disagree = (np.isfinite(b.zbottom) and np.isfinite(b.zbottom_legacy)
                and abs(b.zbottom - b.zbottom_legacy) > _SEABED_XCHECK)
    ok = (np.isfinite(b.zbottom) and not b.is_fallback and b.error <= _BOTTOM_ERR_WARN
          and not disagree)
    if not np.isfinite(b.zbottom):
        note = "no bottom echo found"
    elif b.is_fallback:
        note = (f"near-bottom lower bound (deepest package depth); "
                f"+/- {b.error:.1f} m, {b.n_valid} echoes")
    else:
        note = f"+/- {b.error:.2f} m from {b.n_valid} bottom echoes"
    if disagree:
        note += (f"; stack {b.zbottom:.0f} vs legacy-polyfit {b.zbottom_legacy:.0f} m differ by "
                 f"{abs(b.zbottom - b.zbottom_legacy):.0f} m -- near-bottom multiples (legacy "
                 f"biased deep) or shallow false-lock (stack); inspect")
    return Metric(
        "bottom_depth", round(b.zbottom, 1), "m",
        Status.OK if ok else Status.WARN,
        threshold=_BOTTOM_ERR_WARN,
        source_stage="qa.bottom",
        note=note,
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
