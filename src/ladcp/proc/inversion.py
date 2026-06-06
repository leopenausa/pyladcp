"""Super-ensemble formation and the constrained least-squares inversion.

Port of the core of legacy ``prepinv.m`` (depth-binned super-ensembles with
reference-velocity removal) and ``getinv.m`` (sparse weighted L2 inversion with
smoothing, bottom-track, barotropic and zero-mean constraints).

Velocities are complex ``u + i v``; the model matrix ``G`` is real, so the east and
north components are solved against the same ``G`` with split right-hand sides.
This is a faithful re-implementation, not a line-for-line port; see
docs/VALIDATION_MORIA05.md.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import scipy.sparse as sp
from scipy.sparse.linalg import lsqr

from ..config import CastParams
from ..models import ProfileResult
from .depth import DepthSolution
from .prepare import Ensembles


# --------------------------------------------------------------------------- #
# editing
# --------------------------------------------------------------------------- #
def edit(ens: Ensembles, depth: DepthSolution, params: CastParams,
         zbottom: float | None) -> None:
    """Set weight to NaN for measurements failing edit criteria (in place)."""
    w = ens.weight
    # error-velocity limit
    w[np.abs(ens.re) > params.elim] = np.nan
    # horizontal / vertical velocity limits (gross outliers)
    w[np.abs(ens.ru) > params.vlim] = np.nan
    w[np.abs(ens.rv) > params.vlim] = np.nan
    wm = np.nanmedian(ens.rw[ens.izd[:6]], axis=0)
    w[np.abs(ens.rw - wm[None, :]) > params.wlim + 0.1] = np.nan  # loose vertical gate
    # tilt limit
    bad_tilt = ens.tilt > params.tiltmax[0]
    w[:, bad_tilt] = np.nan
    # below bottom / above surface
    if zbottom is not None and np.isfinite(zbottom):
        w[depth.izm < (-zbottom - 2 * ens.zd[0])] = np.nan
    if ens.izu.size:
        w[depth.izm > -ens.zu[0]] = np.nan


# --------------------------------------------------------------------------- #
# super-ensembles
# --------------------------------------------------------------------------- #
@dataclass
class SuperEns:
    ru: np.ndarray          # [nbin, m]
    rv: np.ndarray
    rw: np.ndarray
    ruvs: np.ndarray        # velocity scatter [nbin, m]
    izm: np.ndarray         # measurement depth [nbin, m] (negative down)
    z: np.ndarray           # instrument depth [m] (negative down)
    time_jul: np.ndarray    # [m]
    dt: np.ndarray          # [m] seconds represented
    slat: np.ndarray
    slon: np.ndarray
    bvel: np.ndarray | None   # [m,4] mean bottom-track
    bvels: np.ndarray | None  # [m] horizontal bottom-track scatter (std)
    wref: np.ndarray          # [m] reference-layer vertical velocity (package W)
    izd: np.ndarray
    izu: np.ndarray


def form_superensembles(ens: Ensembles, depth: DepthSolution, params: CastParams,
                        ctd) -> SuperEns:
    # average over ~one bin length of package depth (legacy p.avdz default):
    # bin spacing is the difference DOWN the bin axis, not along time.
    avdz = float(np.nanmedian(np.abs(np.diff(depth.izm[:, 0]))))
    if not np.isfinite(avdz) or avdz <= 0:
        avdz = params.dz
    izm0 = depth.izm[0]
    nbin, nt = ens.ru.shape

    # reference bins (bins 2,3 of downlooker + the two uplooker bins nearest package)
    izr = list(ens.izd[1:3])
    if ens.izu.size >= 3:
        izr += [ens.izu[-2], ens.izu[-3]]
    izr = np.array(izr, dtype=int)

    cols = []  # list of ensemble-index groups
    ilast = 0
    while ilast < nt - 1:
        rest = np.abs(izm0[ilast + 1:] - izm0[ilast])
        nxt = np.where(rest > avdz)[0]
        step = nxt[0] + 1 if nxt.size else (nt - 1 - ilast)
        i1 = np.arange(ilast, ilast + step + 1)
        half = len(i1) / 2.0 * 1.0  # oversample = 1
        c = int(round(np.mean(i1)))
        i1 = np.arange(c - int(round(half)), c + int(round(half)) + 1)
        i1 = i1[(i1 >= 0) & (i1 < nt)]
        if i1.size == 0:
            break
        cols.append(i1)
        ilast = int(i1.max()) + 1

    m = len(cols)
    su = np.full((nbin, m), np.nan)
    sv = np.full((nbin, m), np.nan)
    sw = np.full((nbin, m), np.nan)
    svs = np.full((nbin, m), np.nan)
    sizm = np.full((nbin, m), np.nan)
    sz = np.full(m, np.nan)
    stime = np.full(m, np.nan)
    sdt = np.full(m, np.nan)
    slat = np.full(m, np.nan)
    slon = np.full(m, np.nan)
    swref = np.full(m, np.nan)  # reference-layer (package) vertical velocity
    sb = np.full((m, 4), np.nan) if ens.bt_vel is not None else None
    sbs = np.full(m, np.nan) if ens.bt_vel is not None else None  # horizontal BT std

    wmask = np.isfinite(ens.weight)
    ru = np.where(wmask, ens.ru, np.nan)
    rv = np.where(wmask, ens.rv, np.nan)
    rw = np.where(wmask, ens.rw, np.nan)

    lat_e = np.interp(_sec(ens.time), ctd.time_elapsed_s + depth.lag_seconds, ctd.lat,
                      left=np.nan, right=np.nan)
    lon_e = np.interp(_sec(ens.time), ctd.time_elapsed_s + depth.lag_seconds, ctd.lon,
                      left=np.nan, right=np.nan)

    for im, i1 in enumerate(cols):
        ur = np.nanmedian(ru[np.ix_(izr, i1)], axis=0)        # package vel per ens
        vr = np.nanmedian(rv[np.ix_(izr, i1)], axis=0)
        wr = np.nanmedian(rw[np.ix_(izr, i1)], axis=0)
        ru_av, rv_av, rw_av = np.nanmean(ur), np.nanmean(vr), np.nanmean(wr)
        ur = np.where(np.isfinite(ur), ur, 0.0)
        vr = np.where(np.isfinite(vr), vr, 0.0)
        wr = np.where(np.isfinite(wr), wr, 0.0)
        su[:, im] = np.nanmedian(ru[:, i1] - ur[None, :], axis=1) + ru_av
        sv[:, im] = np.nanmedian(rv[:, i1] - vr[None, :], axis=1) + rv_av
        sw[:, im] = np.nanmedian(rw[:, i1] - wr[None, :], axis=1) + rw_av
        swref[im] = rw_av
        svs[:, im] = np.sqrt(np.nanstd(ru[:, i1], axis=1) ** 2
                             + np.nanstd(rv[:, i1], axis=1) ** 2)
        sizm[:, im] = np.nanmean(depth.izm[:, i1], axis=1)
        sz[im] = np.nanmean(depth.z[i1])
        stime[im] = np.nanmean(ens.time_jul[i1])
        sdt[im] = i1.size
        slat[im] = np.nanmedian(lat_e[i1])
        slon[im] = np.nanmedian(lon_e[i1])
        if sb is not None:
            sb[im] = np.nanmean(ens.bt_vel[i1], axis=0)
            # horizontal scatter of bottom-track over the super-ensemble (legacy bvels)
            sbs[im] = np.sqrt(np.nanstd(ens.bt_vel[i1, 0]) ** 2
                              + np.nanstd(ens.bt_vel[i1, 1]) ** 2)

    # drop empty super-ensembles
    keep = np.isfinite(np.nanmax(np.where(np.isfinite(su), su, np.nan), axis=0))
    sel = np.where(keep)[0]

    # floor on scatter
    spe = ens.single_ping_err if np.isfinite(ens.single_ping_err) else 0.02
    svs = np.where((svs < spe) | ~np.isfinite(svs), spe, svs)

    if sbs is not None:
        # legacy: single-ping / zero-std bottom tracks -> floor at 0.1 m/s
        sbs = np.where((~np.isfinite(sbs)) | (sbs <= 0), 0.1, sbs)

    dt_sec = np.gradient(stime[sel]) * 86400.0
    return SuperEns(
        ru=su[:, sel], rv=sv[:, sel], rw=sw[:, sel], ruvs=svs[:, sel],
        izm=sizm[:, sel], z=sz[sel], time_jul=stime[sel], dt=np.abs(dt_sec),
        slat=slat[sel], slon=slon[sel],
        bvel=(sb[sel] if sb is not None else None),
        bvels=(sbs[sel] if sbs is not None else None),
        wref=swref[sel],
        izd=ens.izd, izu=ens.izu,
    )


# --------------------------------------------------------------------------- #
# inversion
# --------------------------------------------------------------------------- #
# constraint-normalisation defaults (legacy default.m / getinv.m)
NAV_ERROR = 30.0        # m, p.nav_error
WEIGHTMIN = 0.05        # ps.weightmin
BTRK_RANGE = (50.0, 300.0)   # p.btrk_range [min, max] height above bottom [m]
BTRK_WLIM = 0.05        # p.btrk_wlim, max |W_btrk - W_ref| [m/s]


def _lainsmoo(colsum: np.ndarray, smoofac: float):
    """Port of legacy ``lainsmoo`` (curvature smoothing of one profile dimension).

    ``colsum`` is the accumulated data weight per unknown (``sum|A|`` down each
    column). Returns a list of ``(cols, vals)`` rows penalising curvature, each row
    scaled by the smoothing weight at that column. With ``smoofac==0`` only the
    ill-constrained columns (``< 0.3*median``) receive smoothing — matching the
    legacy "found N ill constrained elements will smooth" behaviour.
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
    rows = []
    if not np.any(w > 0):
        return rows
    cur = np.array([-1.0, 2.0, -1.0])
    # interior curvature rows
    for j in range(1, ls - 1):
        if w[j] > 0:
            rows.append((np.array([j - 1, j, j + 1]), cur * w[j]))
    # edges (legacy 2x first-difference)
    if w[0] > 0:
        rows.append((np.array([0, 1]), np.array([2.0, -2.0]) * w[0]))
    if w[ls - 1] > 0:
        rows.append((np.array([ls - 2, ls - 1]), np.array([-2.0, 2.0]) * w[ls - 1]))
    return rows


@dataclass
class _Solution:
    """One solve of the inverse problem plus the arrays needed to score residuals."""
    uocean: np.ndarray      # complex [nz] ocean velocity
    uctd: np.ndarray        # complex [nt] package/CTD velocity
    velerr: float
    z: np.ndarray           # [nz]
    nz: int
    nt: int
    n_bt: int
    nvel: np.ndarray        # [nz] number of measurements per ocean bin
    d: np.ndarray           # complex [ndata] kept raw data
    jz: np.ndarray          # [ndata] ocean-bin index (0-based)
    iens: np.ndarray        # [ndata] super-ensemble (time) index
    ibin: np.ndarray        # [ndata] super-ensemble bin (row) index


def _solve(se: SuperEns, params: CastParams, *, uship: complex,
           zbottom: float | None, dt_profile: float,
           velerr_override: float | None = None,
           reject: np.ndarray | None = None) -> _Solution:
    """Single weighted L2 inverse solve (port of getinv body up to ``lainsolv``).

    ``velerr_override`` supplies the two-pass velocity error (legacy getinv lines
    105-112). ``reject`` is a boolean ``[nbin, m]`` mask of super-ensemble data
    points to drop (legacy ``lanarrow`` sets ``di.weight=NaN`` on the worst 1%).
    """
    dz = params.dz
    nbin, nt = se.ru.shape
    d = (se.ru + 1j * se.rv).reshape(-1)               # complex data
    izv = (-se.izm).reshape(-1)                        # positive depth of each meas
    jprof = np.repeat(np.arange(nt)[None, :], nbin, axis=0).reshape(-1)
    ibin_full = np.repeat(np.arange(nbin), nt)         # bin row per flattened point
    # velerr (legacy getinv): median over super-ensembles of the across-bin scatter of
    # the VERTICAL velocity in the first downlooker bins, scaled by tan(beam angle).
    nmax = min(se.izd.size, 7)
    sw = np.nanstd(se.rw[se.izd[:nmax]], axis=0)
    sw = sw[sw > 0]
    velerr = max(float(np.nanmedian(sw) / np.tan(np.deg2rad(20.0))), 0.02) if sw.size else 0.02
    if velerr_override is not None and np.isfinite(velerr_override) and velerr_override > 0:
        velerr = float(velerr_override)
    wm = (velerr / se.ruvs).reshape(-1)                # data weight = velerr / scatter

    # depth grid and ocean-bin index per measurement
    jz = np.round(izv / dz).astype(int)
    nz = int(np.nanmax(jz[np.isfinite(jz)])) + 1
    z = (np.arange(1, nz + 1)) * dz

    # legacy edits: drop top/bottom bin, below bottom, and sub-minimum weights
    zb = (zbottom - dz) if zbottom else 1e30
    good = (np.isfinite(d) & np.isfinite(wm) & np.isfinite(izv)
            & (wm >= WEIGHTMIN) & (izv > dz) & (izv < dz * (nz - 1)) & (izv < zb))
    if reject is not None:
        good &= ~reject.reshape(-1)
    d, wm, jz, jprof, ibin = d[good], wm[good], jz[good], jprof[good], ibin_full[good]
    jz = np.clip(jz, 1, nz) - 1                        # 0-based ocean bin
    ndata = d.size

    # weighted data matrices (legacy lainweig: every data row carries its weight wm)
    rows = np.arange(ndata)
    A2 = sp.coo_matrix((wm, (rows, jz)), shape=(ndata, nz)).tocsr()       # ocean
    A1 = sp.coo_matrix((wm, (rows, jprof)), shape=(ndata, nt)).tocsr()    # package/CTD
    rhs = d * wm

    # accumulated data weight per unknown (legacy de.ocean/ctd_constraints)
    range_ocean = np.asarray(np.abs(A2).sum(axis=0)).ravel()    # [nz]
    range_ctd = np.asarray(np.abs(A1).sum(axis=0)).ravel()      # [nt]

    blocks_o, blocks_c, rhs_parts = [A2], [A1], [rhs]

    def add_block(o_block, c_block, r_vals):
        blocks_o.append(o_block)
        blocks_c.append(c_block)
        rhs_parts.append(np.asarray(r_vals, dtype=complex))

    smoofac = float(params.extra.get("smoofac", 0.0))

    # --- smooth ocean velocity profile (curvature on the z grid) ---------------
    srows = _lainsmoo(range_ocean, smoofac)
    if srows:
        nr = len(srows)
        ii = np.concatenate([np.full(c.size, k) for k, (c, _) in enumerate(srows)])
        jj = np.concatenate([c for c, _ in srows])
        vv = np.concatenate([v for _, v in srows])
        add_block(sp.coo_matrix((vv, (ii, jj)), shape=(nr, nz)).tocsr(),
                  sp.csr_matrix((nr, nt)), np.zeros(nr))

    # --- smooth package (CTD) velocity profile in time -------------------------
    crows = _lainsmoo(range_ctd, smoofac)
    if crows:
        nr = len(crows)
        ii = np.concatenate([np.full(c.size, k) for k, (c, _) in enumerate(crows)])
        jj = np.concatenate([c for c, _ in crows])
        vv = np.concatenate([v for _, v in crows])
        add_block(sp.csr_matrix((nr, nz)),
                  sp.coo_matrix((vv, (ii, jj)), shape=(nr, nt)).tocsr(), np.zeros(nr))

    # --- bottom-track constraint (legacy lainbott) -----------------------------
    # Only near-bottom super-ensembles carry usable bottom track: keep those whose
    # height above bottom is within p.btrk_range, weight by velerr/scatter and the
    # sqrt of accumulated package-velocity data weight at that ensemble.
    n_bt = 0
    if (se.bvel is not None and se.bvels is not None and zbottom is not None
            and params.btrk_mode > 0):
        bcomplex = se.bvel[:, 0] + 1j * se.bvel[:, 1]
        hbot = zbottom + se.z                              # height above bottom [m]
        usable = (np.isfinite(bcomplex) & np.isfinite(se.bvels)
                  & (hbot > BTRK_RANGE[0]) & (hbot < BTRK_RANGE[1]))
        # W-difference check (legacy getbtrack): the bottom-track vertical velocity
        # must agree with the reference-layer (package) W, else it is a spurious
        # return — rejects the noisy far-from-bottom pings.
        usable &= np.abs(se.bvel[:, 2] - se.wref) < BTRK_WLIM
        # scatter cut (legacy prepinv btrk_wstd): drop bottom tracks whose horizontal
        # scatter exceeds twice the median of the surviving set.
        if usable.any():
            wstd = 2 * float(np.nanmedian(se.bvels[usable]))
            usable &= se.bvels <= wstd
        idx = np.where(usable)[0]
        if idx.size:
            btweight = velerr / se.bvels[idx]
            botfac = btweight * float(params.extra.get("botfac", 1.0)) * np.sqrt(range_ctd[idx])
            n_bt = idx.size
            cb = sp.coo_matrix((botfac, (np.arange(n_bt), idx)), shape=(n_bt, nt)).tocsr()
            add_block(sp.csr_matrix((n_bt, nz)), cb, bcomplex[idx] * botfac)

    # --- barotropic (GPS) constraint (legacy lainbaro) -------------------------
    barofac = float(params.extra.get("barofac", 1.0))
    have_baro = barofac > 0 and np.isfinite(uship) and dt_profile > 0
    if have_baro:
        barvelerr = 2 * NAV_ERROR / dt_profile
        w = barofac * (velerr / barvelerr)
        # widen tolerance if large data gaps exist
        dt = se.dt
        gaps = dt > 3 * np.nanmean(dt)
        if gaps.sum() > 1:
            facgap = np.nansum(dt[gaps]) / np.nansum(dt)
            w *= (1 - np.tanh(facgap / 0.15))
        if not np.isfinite(w):
            w = 1.0
        fac = np.sqrt(np.nansum(range_ctd))
        crow = (dt / np.nansum(dt) * w * fac)[None, :]
        add_block(sp.csr_matrix((1, nz)), sp.csr_matrix(crow), [-uship * w * fac])

    # --- zero-mean fallback if the mean flow is otherwise unconstrained ---------
    if n_bt == 0 and not have_baro:
        fac = float(np.mean(range_ocean))
        add_block(sp.csr_matrix(np.ones((1, nz)) * fac), sp.csr_matrix((1, nt)), [0.0])

    Ao = sp.vstack(blocks_o).tocsr()
    Ac = sp.vstack(blocks_c).tocsr()
    rhs_full = np.concatenate(rhs_parts)

    G = sp.hstack([Ao, Ac]).tocsr()
    # solve east and north against the same G
    m_re = lsqr(G, np.real(rhs_full), atol=1e-9, btol=1e-9, iter_lim=8000)[0]
    m_im = lsqr(G, np.imag(rhs_full), atol=1e-9, btol=1e-9, iter_lim=8000)[0]
    uocean = m_re[:nz] + 1j * m_im[:nz]
    uctd = m_re[nz:nz + nt] + 1j * m_im[nz:nz + nt]
    nvel = np.asarray(A2.astype(bool).sum(axis=0)).ravel()

    return _Solution(uocean=uocean, uctd=uctd, velerr=velerr, z=z, nz=nz, nt=nt,
                     n_bt=n_bt, nvel=nvel, d=d, jz=jz, iens=jprof, ibin=ibin)


def _ocean_error(sol: _Solution) -> np.ndarray:
    """Per-ocean-bin velocity error (port of geterr ``u_oce_s``/``v_oce_s``).

    At each output depth bin, ``u_oce = measured - fitted package velocity`` is an
    independent estimate of the ocean velocity from every contributing measurement;
    its scatter across measurements is the velocity error. Returns
    ``sqrt(std(u_oce)^2 + std(v_oce)^2)`` per bin.
    """
    ocean_est = sol.d - sol.uctd[sol.iens]             # complex per kept point
    uerr = np.full(sol.nz, np.nan)
    for k in range(sol.nz):
        m = sol.jz == k
        if m.sum() >= 2:
            uerr[k] = np.hypot(np.std(np.real(ocean_est[m])),
                               np.std(np.imag(ocean_est[m])))
    return uerr


def _lanarrow_reject(sol: _Solution, shape: tuple[int, int],
                     frac: float = 0.01) -> np.ndarray:
    """Mask of the worst-``frac`` super-ensemble data points by fit residual.

    Port of legacy ``lanarrow``: residual = ``|d - (u_ocean[jz] + u_ctd[iens])|``;
    the largest 1% are flagged so the next solve drops them.
    """
    reject = np.zeros(shape, dtype=bool)
    resid = sol.d - (sol.uocean[sol.jz] + sol.uctd[sol.iens])
    dif = np.abs(resid) ** 2
    ok = np.where(np.isfinite(dif))[0]
    if ok.size == 0:
        return reject
    n = int(ok.size * frac)
    if n < 1:
        return reject
    worst = ok[np.argsort(dif[ok])[-n:]]
    reject[sol.ibin[worst], sol.iens[worst]] = True
    return reject


def invert(se: SuperEns, params: CastParams, *, lat: float, lon: float,
           uship: complex, zbottom: float | None, station: str,
           declination: float, dt_profile: float) -> ProfileResult:
    """Two-pass constrained inversion (process_cast STEP 11 + STEP 14).

    Pass 1 uses the super-ensemble-scatter velocity error. Its residuals drive
    (a) the ``lanarrow`` rejection of the worst 1% of super-ensemble data and
    (b) the two-pass velocity error ``velerr = median(uerr)``. Pass 2 re-solves
    with that error on the cleaned data. ``params.outlier <= 0`` reverts to a
    single pass.
    """
    dz = params.dz
    sol = _solve(se, params, uship=uship, zbottom=zbottom, dt_profile=dt_profile)

    n_outlier = int(np.ceil(params.outlier)) if params.outlier and params.outlier > 0 else 0
    reject = None
    if n_outlier > 0:
        # legacy lanarrow loops `outlier` times, accumulating 1% rejections; the
        # residual-based velerr (set after the first solve) feeds every later solve.
        velerr2 = float(np.nanmedian(_ocean_error(sol)))
        for _ in range(n_outlier):
            reject = _lanarrow_reject(sol, se.ru.shape, frac=0.01) if reject is None \
                else (reject | _lanarrow_reject(sol, se.ru.shape, frac=0.01))
            sol = _solve(se, params, uship=uship, zbottom=zbottom, dt_profile=dt_profile,
                         velerr_override=velerr2, reject=reject)

    uerr = _ocean_error(sol)
    uerr = np.where(np.isfinite(uerr), uerr, sol.velerr)

    pr = ProfileResult(
        station=station, name=params.cruise_id + " " + station,
        lat=lat, lon=lon, magnetic_deviation=declination,
        z=sol.z, u=np.real(sol.uocean), v=np.imag(sol.uocean),
        uerr=uerr, nvel=sol.nvel,
        ubar=float(np.nanmean(np.real(sol.uocean))),
        vbar=float(np.nanmean(np.imag(sol.uocean))),
        bottom_depth=(float(zbottom) if zbottom else np.nan),
        config={"dz": dz, "nz": sol.nz, "n_superens": sol.nt, "declination": declination,
                "velerr": sol.velerr, "n_bottom_track": sol.n_bt,
                "n_outlier_passes": n_outlier},
    )
    return pr


def _sec(t):
    return (t - t[0]) / np.timedelta64(1, "s")
