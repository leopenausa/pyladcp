"""Full sparse least-squares inverse (``getinv.m``) — the ``solver="inverse"`` path.

The shear solver (:mod:`ladcp.qa.inverse`) reproduces the golden ``ps.shear==1`` product
by taking the shear method's baroclinic *shape* and adding a barotropic reference. This
module instead forms the **constrained weighted L2 inverse** that legacy ``getinv.m``
solves: every super-ensemble cell is one data row ``u_ocean(z) + u_ctd(t) = measured``,
augmented with curvature smoothing, a bottom-track constraint (``lainbott``), a barotropic
navigation constraint (``lainbaro``) and -- when present -- a ship-ADCP constraint
(``lainsadcp``, added in #5). The east/north components share one real model matrix ``G``
and are solved with complex right-hand sides.

This is a port of the (now-removed) ``proc/inversion.py`` onto the live ``qa.SuperEns``
data model. Two conventions differ from the old code and are handled here: ``izm``/``z``
are **positive-down** (the old solver used negative-down), and the per-super-ensemble
bottom-track / navigation auxiliaries are *derived* (:func:`inverse_inputs`) rather than
carried on the super-ensemble, because ``qa`` keeps bottom-track per-ping.

The fidelity divergences catalogued in the legacy-vs-python audit are kept ON by default:
the two-pass ``velerr`` override and the ``lanarrow`` 1% outlier loop (``outlier`` passes).
"""

from __future__ import annotations

import warnings
from dataclasses import dataclass, replace

import numpy as np
import scipy.sparse as sp
from scipy.sparse.linalg import lsqr

from ..models import CTDTimeSeries
from .depth import SyncResult, water_window
from .inverse import _SURFACE_FILL_MAX, VelocityProfile
from .superens import SuperEns, _uvrot

# constraint-normalisation defaults (legacy default.m / getinv.m)
NAV_ERROR = 30.0              # m, p.nav_error (GPS fix uncertainty)
BTRK_RANGE = (50.0, 300.0)    # p.btrk_range: package height above bottom to trust BT [m]


# --------------------------------------------------------------------------- #
# per-super-ensemble auxiliaries (bottom-track + navigation)
# --------------------------------------------------------------------------- #
@dataclass
class InverseAux:
    """Per-super-ensemble constraint inputs derived from per-ping products."""

    bvel: np.ndarray | None     # [nse] complex bottom-track reference (-u_package), or None
    bvels: np.ndarray | None    # [nse] horizontal bottom-track scatter [m/s]
    hbot: np.ndarray | None     # [nse] package height above seabed [m]
    uship: complex              # depth-mean ship velocity over ground (u+iv) [m/s]
    dt: np.ndarray              # [nse] seconds represented by each super-ensemble
    dt_profile: float           # total in-water cast duration [s]


def inverse_inputs(se: SuperEns, *, bt=None, sync: SyncResult | None = None,
                   ctd: CTDTimeSeries | None = None, ping_dt: float = 1.0,
                   zbottom: float = np.nan) -> InverseAux:
    """Collapse per-ping bottom-track + nav onto the super-ensemble columns.

    ``bt`` is the per-ping :class:`~ladcp.qa.bottom.BottomTrack` (already W-checked and
    outlier-cleaned). For each kept super-ensemble (``se.group_pings``) the bottom-track
    reference is the mean of its pings' ``bvel`` and the horizontal scatter ``bvels`` is
    floored at 0.1 m/s (legacy single-ping default). ``uship`` and ``dt_profile`` come
    from the CTD navigation over the in-water window; ``dt`` is each super-ensemble's
    duration in seconds (``ping count * ping_dt``).
    """
    nse = se.n_se
    groups = se.group_pings
    dt = (se.dtiv.astype(float) * ping_dt)

    bvel = bvels = hbot = None
    if bt is not None and groups is not None:
        bvel = np.full(nse, np.nan, dtype=complex)
        bvels = np.full(nse, np.nan)
        for i, g in enumerate(groups):
            b = bt.bvel[g]
            if np.isfinite(b).sum() >= 1:
                bvel[i] = np.nanmean(b)
                bvels[i] = float(np.hypot(np.nanstd(np.real(b)), np.nanstd(np.imag(b))))
        # single-ping / zero-scatter bottom tracks -> floor (legacy 0.1 m/s)
        bvels = np.where(~np.isfinite(bvels) | (bvels <= 0), 0.1, bvels)
        bvels = np.where(np.isfinite(bvel), bvels, np.nan)
        if np.isfinite(zbottom):
            hbot = zbottom - se.z                       # package height above seabed [m]

    uship = 0 + 0j
    dt_profile = float(np.nansum(dt))
    if sync is not None and ctd is not None:
        uship, dt_profile = _ship_velocity(se, sync, ctd, ping_dt)
    return InverseAux(bvel=bvel, bvels=bvels, hbot=hbot, uship=uship, dt=dt,
                      dt_profile=dt_profile)


def _ship_velocity(se: SuperEns, sync: SyncResult, ctd: CTDTimeSeries,
                   ping_dt: float) -> tuple[complex, float]:
    """Depth-mean ship velocity over ground (u+iv) and cast duration [s].

    Ship position per ping is the CTD navigation interpolated onto the ping's CTD-time
    (``ping_elapsed - lag``); displacement over the in-water window divided by its
    duration gives the barotropic ``uship`` the navigation constraint pins to.
    """
    n = sync.z_on_ping.size
    tad = np.arange(n, dtype=float) * ping_dt           # ping elapsed time [s]
    i0, i1 = water_window(sync.z_on_ping)
    dt_profile = float(tad[i1] - tad[i0])
    if dt_profile <= 0:
        return 0 + 0j, max(dt_profile, 1.0)
    tctd = tad - float(sync.lag)                         # ping time on the CTD clock
    lat = np.interp(tctd, ctd.time_elapsed_s, ctd.lat)
    lon = np.interp(tctd, ctd.time_elapsed_s, ctd.lon)
    latm = float(np.nanmedian(lat))
    x = (lon[i1] - lon[i0]) * 111320.0 * np.cos(np.deg2rad(latm))   # east displacement [m]
    y = (lat[i1] - lat[i0]) * 110540.0                              # north displacement [m]
    return complex(x / dt_profile, y / dt_profile), dt_profile


# --------------------------------------------------------------------------- #
# smoothing (legacy lainsmoo)
# --------------------------------------------------------------------------- #
def _lainsmoo(colsum: np.ndarray, smoofac: float) -> list[tuple[np.ndarray, np.ndarray]]:
    """Curvature-smoothing rows for one profile dimension (port of ``lainsmoo``).

    ``colsum`` is the accumulated data weight per unknown (``sum|A|`` down each column).
    With ``smoofac==0`` only the ill-constrained columns (``< 0.3*median``) are smoothed,
    matching the legacy "found N ill constrained elements will smooth" branch.
    """
    ls = colsum.size
    fs = np.sqrt(np.maximum(colsum, 0.0))
    fsm = max(float(np.median(fs)), 0.01)
    ibad = np.where(fs < fsm * 0.3)[0]
    fs = np.maximum(fs, fsm * 0.1)
    fs1 = fsm / fs
    w = fs1 * smoofac
    if ibad.size:
        w[ibad] = np.maximum(fs1[ibad], 0.5)
    rows: list[tuple[np.ndarray, np.ndarray]] = []
    if not np.any(w > 0):
        return rows
    cur = np.array([-1.0, 2.0, -1.0])
    for j in range(1, ls - 1):
        if w[j] > 0:
            rows.append((np.array([j - 1, j, j + 1]), cur * w[j]))
    if w[0] > 0:
        rows.append((np.array([0, 1]), np.array([2.0, -2.0]) * w[0]))
    if w[ls - 1] > 0:
        rows.append((np.array([ls - 2, ls - 1]), np.array([-2.0, 2.0]) * w[ls - 1]))
    return rows


def _smooth_block(rows, nz, nt, *, on_ocean):
    """Pack ``_lainsmoo`` rows into (ocean_block, ctd_block) sparse matrices."""
    nr = len(rows)
    ii = np.concatenate([np.full(c.size, k) for k, (c, _) in enumerate(rows)])
    jj = np.concatenate([c for c, _ in rows])
    vv = np.concatenate([v for _, v in rows])
    width = nz if on_ocean else nt
    block = sp.coo_matrix((vv, (ii, jj)), shape=(nr, width)).tocsr()
    if on_ocean:
        return block, sp.csr_matrix((nr, nt))
    return sp.csr_matrix((nr, nz)), block


def _lainsadcp(svel: np.ndarray, nz: int, dz: float, sadcpfac: float, velerr: float):
    """Ship-ADCP ocean-velocity constraint rows (port of ``getinv.m`` ``lainsadcp``).

    ``svel`` is the SADCP profile matched to the cast, columns ``(z, u, v, verr)`` in the
    **magnetic** frame (same as the LADCP data). Each finite point pins the ocean velocity
    at its depth bin: row weight ``sadcpfac * velerr/verr`` (legacy halving/column-norm
    ``fac2`` is computed but commented out in the source, so it is omitted here -- this
    reproduces the logged "mean sadcp weight" ~1.6 on CRUISE2_002). Only the ocean block is
    touched (``Ac`` row = 0). Returns ``(jz, w, rhs, kept)`` where ``kept`` is the
    ``fac>0.1`` mask the legacy stores as ``dr.*_sadcp``; empty arrays if unusable.

    Validated end-to-end with the VmDAS ingester (:mod:`ladcp.io.sadcp_vmdas`): on MORIA
    79/80/82 the constrained inverse and the absolute shipboard-ADCP profile agree to
    0.015-0.05 m/s over the upper 300 m.
    """
    z = np.abs(svel[:, 0])
    d = svel[:, 1] + 1j * svel[:, 2]
    verr = svel[:, 3].astype(float).copy()
    ok = np.isfinite(d)
    z, d, verr = z[ok], d[ok], verr[ok]
    empty = (np.array([], int), np.array([]), np.array([], complex), np.zeros(0, bool))
    if z.size < 1 or sadcpfac <= 0:
        return empty
    bad = ~np.isfinite(verr) | (verr == 0)
    if (~bad).any():
        verr[bad] = float(np.nanmax(verr[~bad]))
    else:
        verr[:] = 1.0
    if z.size > 3:                                      # weight by scatter when >1 profile
        if not (velerr > 0):
            s = np.sort(verr)
            velerr = float(np.mean(s[:int(np.ceil(0.2 * s.size))]))
        fac = velerr / verr
    else:
        fac = np.ones(z.size)
    jz = np.clip(np.round(z / dz).astype(int), 1, nz) - 1
    w = sadcpfac * fac
    return jz, w, d * w, (fac > 0.1)


# --------------------------------------------------------------------------- #
# one weighted L2 solve
# --------------------------------------------------------------------------- #
@dataclass
class ConstraintWeights:
    """Per-constraint accumulated weights of the inverse (legacy Figure 12).

    For each constraint block the column sums ``sum|w|`` over the ocean unknowns
    (``ocean[name][nz]``, vs depth ``z``) and the reference/CTD unknowns
    (``ctd[name][nt]``, vs super-ensemble index) — the stacked bars of the legacy
    "ocean / CTD velocity constraints" panels.
    """

    z: np.ndarray                       # [nz] ocean-bin depth [m, +down]
    ocean: dict[str, np.ndarray]        # constraint name -> [nz] sum of |weights|
    ctd: dict[str, np.ndarray]          # constraint name -> [nt] sum of |weights|


@dataclass
class _Solution:
    uocean: np.ndarray          # complex [nz] ocean velocity (magnetic frame)
    uctd: np.ndarray            # complex [nt] reference (-package) velocity
    velerr: float
    z: np.ndarray               # [nz] depth grid [m, +down]
    nz: int
    nt: int
    n_bt: int
    nvel: np.ndarray            # [nz] measurements per ocean bin
    d: np.ndarray               # complex [ndata] kept data
    jz: np.ndarray              # [ndata] ocean-bin index
    iens: np.ndarray            # [ndata] super-ensemble index
    ibin: np.ndarray            # [ndata] super-ensemble bin (row) index
    n_sadcp: int = 0            # SADCP constraint rows applied
    weights: ConstraintWeights | None = None


def _solve(se: SuperEns, aux: InverseAux, *, dz: float, weightmin: float,
           smoofac: float, botfac: float, barofac: float, zbottom: float,
           svel: np.ndarray | None = None, sadcpfac: float = 0.0,
           velerr_override: float | None = None,
           reject: np.ndarray | None = None) -> _Solution:
    """Single constrained weighted L2 inverse solve (``getinv`` body up to ``lainsolv``).

    Data rows model ``measured = u_ocean(z) + u_ctd(t)``; ``u_ctd`` is the reference
    (negative package) velocity. ``svel`` is an optional magnetic-frame SADCP profile
    ``(z, u, v, verr)`` whose rows pin the ocean velocity (``lainsadcp``).
    ``velerr_override`` supplies the two-pass velocity error; ``reject`` is a
    ``[nbin, nse]`` mask of ``lanarrow``-flagged points to drop.
    """
    nbin, nt = se.ru.shape
    d = (se.ru + 1j * se.rv).reshape(-1)               # complex data (magnetic)
    izv = se.izm.reshape(-1)                           # +down measurement depth
    jprof = np.repeat(np.arange(nt)[None, :], nbin, axis=0).reshape(-1)
    ibin_full = np.repeat(np.arange(nbin), nt)

    # velerr (legacy getinv): median over super-ensembles of the across-bin scatter of the
    # vertical velocity in the first downlooker bins, scaled by tan(beam angle).
    nmax = min(se.izd.size, 7)
    with warnings.catch_warnings():
        warnings.simplefilter("ignore", RuntimeWarning)   # DOF<=0 on sparse SE columns
        sw = np.nanstd(se.rw[se.izd[:nmax]], axis=0)
    sw = sw[sw > 0]
    velerr = max(float(np.nanmedian(sw) / np.tan(np.deg2rad(20.0))), 0.02) if sw.size else 0.02
    if velerr_override is not None and np.isfinite(velerr_override) and velerr_override > 0:
        velerr = float(velerr_override)
    wm = (velerr / se.ruvs).reshape(-1)                # data weight = velerr / scatter

    jz = np.round(izv / dz).astype(float)
    finite_jz = np.isfinite(jz)
    nz = int(np.nanmax(jz[finite_jz])) + 1
    if np.isfinite(zbottom):                           # never grid below the seabed
        nz = min(nz, int(np.floor(zbottom / dz)))
    z = np.arange(1, nz + 1) * dz

    zb = (zbottom - dz) if np.isfinite(zbottom) else 1e30
    good = (np.isfinite(d) & np.isfinite(wm) & finite_jz
            & (wm >= weightmin) & (izv > dz) & (izv < dz * (nz - 1)) & (izv < zb))
    if reject is not None:
        good &= ~reject.reshape(-1)
    jz = np.where(finite_jz, jz, 0).astype(int)
    d, wm, jz, jprof, ibin = d[good], wm[good], jz[good], jprof[good], ibin_full[good]
    jz = np.clip(jz, 1, nz) - 1                         # 0-based ocean bin
    ndata = d.size

    rows = np.arange(ndata)
    A2 = sp.coo_matrix((wm, (rows, jz)), shape=(ndata, nz)).tocsr()       # ocean
    A1 = sp.coo_matrix((wm, (rows, jprof)), shape=(ndata, nt)).tocsr()    # reference/CTD
    rhs = d * wm
    range_ocean = np.asarray(np.abs(A2).sum(axis=0)).ravel()
    range_ctd = np.asarray(np.abs(A1).sum(axis=0)).ravel()

    blocks_o, blocks_c, rhs_parts = [A2], [A1], [rhs]
    block_names = ["velocity"]

    def add_block(o_block, c_block, r_vals, name):
        blocks_o.append(o_block)
        blocks_c.append(c_block)
        rhs_parts.append(np.asarray(r_vals, dtype=complex))
        block_names.append(name)

    srows = _lainsmoo(range_ocean, smoofac)
    if srows:
        ob, cb = _smooth_block(srows, nz, nt, on_ocean=True)
        add_block(ob, cb, np.zeros(len(srows)), "smoothing")
    crows = _lainsmoo(range_ctd, smoofac)
    if crows:
        ob, cb = _smooth_block(crows, nz, nt, on_ocean=False)
        add_block(ob, cb, np.zeros(len(crows)), "smoothing")

    # --- bottom-track constraint (legacy lainbott) ----------------------------- #
    n_bt = 0
    if (aux.bvel is not None and aux.bvels is not None and aux.hbot is not None
            and botfac > 0):
        usable = (np.isfinite(aux.bvel) & np.isfinite(aux.bvels)
                  & (aux.hbot > BTRK_RANGE[0]) & (aux.hbot < BTRK_RANGE[1]))
        if usable.any():
            wstd = 2 * float(np.nanmedian(aux.bvels[usable]))
            usable &= aux.bvels <= wstd
        idx = np.where(usable)[0]
        if idx.size:
            btw = (velerr / aux.bvels[idx]) * botfac * np.sqrt(range_ctd[idx])
            n_bt = idx.size
            cb = sp.coo_matrix((btw, (np.arange(n_bt), idx)), shape=(n_bt, nt)).tocsr()
            add_block(sp.csr_matrix((n_bt, nz)), cb, aux.bvel[idx] * btw, "bottom track")

    # --- barotropic navigation constraint (legacy lainbaro) -------------------- #
    have_baro = barofac > 0 and np.isfinite(aux.uship) and aux.dt_profile > 0
    if have_baro:
        barvelerr = 2 * NAV_ERROR / aux.dt_profile
        w = barofac * (velerr / barvelerr)
        dt = aux.dt
        gaps = dt > 3 * np.nanmean(dt)
        if gaps.sum() > 1:
            facgap = np.nansum(dt[gaps]) / np.nansum(dt)
            w *= (1 - np.tanh(facgap / 0.15))
        if not np.isfinite(w):
            w = 1.0
        fac = np.sqrt(np.nansum(range_ctd))
        crow = (dt / np.nansum(dt) * w * fac)[None, :]
        # uctd is the reference (-package) velocity, so the depth-mean is pinned to -uship
        # (legacy lainbaro RHS is -1*uship). The earlier +uship was silent on station-kept casts
        # (uship~0) but flipped the barotropic by 2*uship on a steaming ship (MORIA-20).
        add_block(sp.csr_matrix((1, nz)), sp.csr_matrix(crow), [-aux.uship * w * fac],
                  "GPS navigation")

    # --- ship-ADCP constraint (legacy lainsadcp) ------------------------------- #
    n_sadcp = 0
    if svel is not None and sadcpfac > 0 and np.isfinite(svel[:, 1]).sum() > 2:
        jzs, ws, rhss, _ = _lainsadcp(svel, nz, dz, sadcpfac, velerr)
        if jzs.size:
            n_sadcp = jzs.size
            ob = sp.coo_matrix((ws, (np.arange(n_sadcp), jzs)),
                               shape=(n_sadcp, nz)).tocsr()
            add_block(ob, sp.csr_matrix((n_sadcp, nt)), rhss, "ship ADCP")

    # --- zero-mean fallback (legacy lainocean) --------------------------------- #
    if n_bt == 0 and not have_baro and n_sadcp == 0:
        fac = float(np.mean(range_ocean))
        add_block(sp.csr_matrix(np.ones((1, nz)) * fac), sp.csr_matrix((1, nt)), [0.0],
                  "zero-mean")

    wsum_o: dict[str, np.ndarray] = {}
    wsum_c: dict[str, np.ndarray] = {}
    for name, ob, cb in zip(block_names, blocks_o, blocks_c, strict=True):
        so = np.asarray(np.abs(ob).sum(axis=0)).ravel()
        sc = np.asarray(np.abs(cb).sum(axis=0)).ravel()
        wsum_o[name] = wsum_o.get(name, 0.0) + so
        wsum_c[name] = wsum_c.get(name, 0.0) + sc
    weights = ConstraintWeights(z=z, ocean=wsum_o, ctd=wsum_c)

    G = sp.hstack([sp.vstack(blocks_o), sp.vstack(blocks_c)]).tocsr()
    rhs_full = np.concatenate(rhs_parts)
    m_re = lsqr(G, np.real(rhs_full), atol=1e-9, btol=1e-9, iter_lim=8000)[0]
    m_im = lsqr(G, np.imag(rhs_full), atol=1e-9, btol=1e-9, iter_lim=8000)[0]
    uocean = m_re[:nz] + 1j * m_im[:nz]
    uctd = m_re[nz:nz + nt] + 1j * m_im[nz:nz + nt]
    nvel = np.asarray(A2.astype(bool).sum(axis=0)).ravel()

    return _Solution(uocean=uocean, uctd=uctd, velerr=velerr, z=z, nz=nz, nt=nt,
                     n_bt=n_bt, nvel=nvel, d=d, jz=jz, iens=jprof, ibin=ibin,
                     n_sadcp=n_sadcp, weights=weights)


def _ocean_error(sol: _Solution) -> np.ndarray:
    """Per-ocean-bin velocity error (port of ``geterr`` u_oce_s/v_oce_s).

    ``u_ocean = measured - fitted reference`` is an independent ocean estimate from every
    contributing measurement; its scatter across measurements is the per-bin error.
    """
    ocean_est = sol.d - sol.uctd[sol.iens]
    uerr = np.full(sol.nz, np.nan)
    for k in range(sol.nz):
        m = sol.jz == k
        if m.sum() >= 2:
            uerr[k] = float(np.hypot(np.std(np.real(ocean_est[m])),
                                     np.std(np.imag(ocean_est[m]))))
    return uerr


def _lanarrow_reject(sol: _Solution, shape: tuple[int, int],
                     frac: float = 0.01) -> np.ndarray:
    """Mask of the worst-``frac`` super-ensemble points by fit residual (``lanarrow``)."""
    reject = np.zeros(shape, dtype=bool)
    dif = np.abs(sol.d - (sol.uocean[sol.jz] + sol.uctd[sol.iens])) ** 2
    ok = np.where(np.isfinite(dif))[0]
    n = int(ok.size * frac)
    if n < 1:
        return reject
    worst = ok[np.argsort(dif[ok])[-n:]]
    reject[sol.ibin[worst], sol.iens[worst]] = True
    return reject


def _surface_fill(u: np.ndarray, v: np.ndarray, uerr: np.ndarray, nvel: np.ndarray,
                  z: np.ndarray, *, frac: float = 0.4, floor: float = 3.0) -> None:
    """Fill the under-constrained surface bins from the first reliable bin (in place).

    The shallowest ocean bins of the inverse are weakly constrained and confounded with
    the package velocity, so they collapse toward zero (diagnosed on MORIA-79 v). As the
    shear path does (:func:`~ladcp.qa.inverse.velocity_profile`), every bin above the
    shallowest *reliable* bin is set to that bin's value -- but ONLY across a genuine surface
    gap (the first reliable bin shallower than ``_SURFACE_FILL_MAX``). A larger gap means the
    upper water column was never sampled (e.g. an ADCP file that began mid-descent on a
    fragmented cast); it is left NaN rather than padded with a fake constant. The coverage
    threshold is higher than the shear path's (``0.15*median``) because the inverse's collapse
    spans several bins rather than the odd sparse one, so a single-bin criterion would leave it
    half-fixed.
    """
    good = nvel[nvel > 0]
    if good.size == 0:
        return
    nmed = float(np.median(good))
    reliable = nvel >= max(floor, frac * nmed)
    if not reliable.any():
        return
    lo = int(np.argmax(reliable))
    if z[lo] > _SURFACE_FILL_MAX:                          # large unsampled gap -> honest NaN
        samp = np.where(nvel > 0)[0]                       # NaN the smoothed top above real data
        cut = int(samp[0]) if samp.size else lo
        for a in (u, v, uerr):
            a[:cut] = np.nan
        return
    for a in (u, v, uerr):
        a[:lo] = a[lo]


# --------------------------------------------------------------------------- #
# two-pass driver
# --------------------------------------------------------------------------- #
def invert(se: SuperEns, aux: InverseAux, *, dz: float = 8.0, drot: float = 0.0,
           zbottom: float = np.nan, weightmin: float = 0.05, smoofac: float = 0.0,
           botfac: float = 1.0, barofac: float = 1.0, outlier: float = 1.0,
           svel: np.ndarray | None = None, sadcpfac: float = 0.0,
           diag: dict | None = None) -> VelocityProfile:
    """Two-pass constrained inverse (process_cast STEP 11 + STEP 14) -> profile.

    Pass 1 uses the super-ensemble-scatter velocity error. Its residuals drive both the
    ``lanarrow`` rejection of the worst 1% of data and the two-pass velocity error
    ``velerr = median(uerr)``; pass 2 re-solves with that error on the cleaned data.
    ``outlier <= 0`` reverts to a single pass. ``svel`` is an optional ship-ADCP profile
    ``(z, u, v, verr)`` in the **true** frame; it is rotated into the magnetic solve frame
    and added as the ``lainsadcp`` ocean constraint with weight ``sadcpfac``. The ocean
    velocity is rotated magnetic->true by ``drot``.

    ``diag``, if a dict is passed, receives ``weights``: the final pass's
    :class:`ConstraintWeights` (legacy Figure 12 stacked-bar data).
    """
    if se.ru.shape[1] == 0:                              # no super-ensembles (degenerate cast)
        z = np.arange(1, 2) * dz                         # -> empty NaN profile, not a crash
        nan = np.full(z.size, np.nan)
        return VelocityProfile(z=z, u=nan, v=nan, uerr=nan, nvel=np.zeros(z.size, dtype=int),
                               ubar=np.nan, vbar=np.nan, n=np.zeros(z.size, dtype=int))

    svel_mag = None
    if svel is not None and sadcpfac > 0:
        svel_mag = np.asarray(svel, dtype=float).copy()
        su, sv = _uvrot(svel_mag[:, 1], svel_mag[:, 2], -drot)     # true -> magnetic
        svel_mag[:, 1], svel_mag[:, 2] = su, sv

    # The GPS ship velocity (aux.uship) is built in the TRUE (geographic) frame, but the solve
    # data/unknowns are MAGNETIC (se.ru/rv are rotated to true only at the end). Rotate uship
    # true->magnetic so the lainbaro RHS is frame-consistent -- the same handling the SADCP
    # constraint gets above. Legacy rotates all data to true at load (loadnav.m:178-179), so its
    # lainbaro uship is already frame-consistent; pyladcp must rotate here. Error w/o this is
    # ~|uship|*drot_rad: negligible on station-kept casts, real on drifting/steaming ones.
    if drot and np.isfinite(aux.uship):
        su, sv = _uvrot(aux.uship.real, aux.uship.imag, -drot)    # true -> magnetic
        aux = replace(aux, uship=complex(su, sv))

    if diag is not None:
        diag["uship"] = aux.uship

    def solve(**kw):
        return _solve(se, aux, dz=dz, weightmin=weightmin, smoofac=smoofac, botfac=botfac,
                      barofac=barofac, zbottom=zbottom, svel=svel_mag, sadcpfac=sadcpfac,
                      **kw)

    sol = solve()
    n_pass = int(np.ceil(outlier)) if outlier and outlier > 0 else 0
    reject = None
    if n_pass > 0:
        velerr2 = float(np.nanmedian(_ocean_error(sol)))
        for _ in range(n_pass):
            rj = _lanarrow_reject(sol, se.ru.shape, frac=0.01)
            reject = rj if reject is None else (reject | rj)
            sol = solve(velerr_override=velerr2, reject=reject)

    if diag is not None:
        diag["weights"] = sol.weights
    uerr = _ocean_error(sol)
    uerr = np.where(np.isfinite(uerr), uerr, sol.velerr)
    u, v = _uvrot(np.real(sol.uocean), np.imag(sol.uocean), drot)   # magnetic -> true

    # trim the trailing unconstrained tail (empty bins between the deepest data and the
    # seabed) so the profile ends at real data, and average the reference over data bins.
    z, nvel = sol.z, sol.nvel
    has = np.where(nvel > 0)[0]
    if has.size:
        last = int(has[-1]) + 1
        z, u, v, uerr, nvel = z[:last], u[:last], v[:last], uerr[:last], nvel[:last]
    _surface_fill(u, v, uerr, nvel, z)                 # repair the collapsed surface bins
    valid = nvel > 0
    return VelocityProfile(
        z=z, u=u, v=v, uerr=uerr, nvel=nvel,
        ubar=float(np.nanmean(u[valid])), vbar=float(np.nanmean(v[valid])),
        n=nvel.copy(),
    )
