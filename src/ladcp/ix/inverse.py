"""Constrained least-squares inverse for the ocean velocity profile (getinv.m).

The LADCP inverse solves for two unknown profiles at once: the ocean velocity on a fixed
depth grid (``uocean[nz]``) and the package/CTD velocity per super-ensemble (``uctd[nt]``).
Each measured super-ensemble velocity is modelled as ``uocean[zbin] + uctd[ens]``; that data
block is stacked with curvature-smoothing rows (both profiles), a bottom-track block tying
near-bottom package velocity to the seabed-referenced absolute velocity, and a barotropic
(GPS) row tying the time-weighted mean package velocity to the ship drift. The over-determined
complex system is solved in one weighted L2 pass (``lsqr``) for east+north.

Faithful to ``getinv.m`` (``lainweig``/``lainsmoo``/``lainbott``/``lainbaro``) and the two-pass
velocity-error + 1% ``lanarrow`` outlier loop of ``process_cast`` STEP 11/14. Consumes the
:class:`ladcp.ix.superens.SuperEns` directly. SADCP and surface-drag constraints are out of
scope here (SADCP is a later phase); the Nav+bottom-track core is validated on quiet stations.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any

import numpy as np
import scipy.sparse as sp
from scipy.sparse.linalg import lsqr

from .superens import SuperEns

NAV_ERROR = 30.0        # p.nav_error [m] GPS fix uncertainty
WEIGHTMIN = 0.05        # ps.weightmin, drop data below this weight
BTRK_RANGE = (50.0, 300.0)   # p.btrk_range [min, max] height above bottom [m]
BTRK_WLIM = 0.05        # p.btrk_wlim, max |W_btrk - W_ref| [m/s]
BEAM_ANGLE = 20.0


@dataclass
class InverseResult:
    """Final ocean velocity profile (≈ legacy ``dr``)."""

    z: np.ndarray            # [nz] depth grid (m, positive down)
    u: np.ndarray            # [nz] east ocean velocity (m/s)
    v: np.ndarray            # [nz] north ocean velocity
    uerr: np.ndarray         # [nz] velocity error
    nvel: np.ndarray         # [nz] measurements per bin
    ubar: float
    vbar: float
    velerr: float
    uctd: np.ndarray         # [nt] complex package velocity
    n_bt: int
    config: dict[str, Any] = field(default_factory=dict)
    meta: dict[str, Any] = field(default_factory=dict)


def _lainsmoo(colsum: np.ndarray, smoofac: float) -> list[tuple[np.ndarray, np.ndarray]]:
    """Curvature-smoothing rows for one profile dimension (legacy ``lainsmoo``).

    ``colsum`` is the accumulated data weight per unknown. Each interior unknown gets a
    ``[-1,2,-1]`` curvature row scaled by the local smoothing weight; edges get a doubled
    first difference. With ``smoofac==0`` only ill-constrained columns (``<0.3*median``)
    are smoothed (the legacy "ill constrained elements" fallback).
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


@dataclass
class _Solution:
    uocean: np.ndarray
    uctd: np.ndarray
    velerr: float
    z: np.ndarray
    nz: int
    nt: int
    n_bt: int
    nvel: np.ndarray
    d: np.ndarray
    jz: np.ndarray
    iens: np.ndarray
    ibin: np.ndarray


def _solve(se: SuperEns, *, uship: complex, zbottom: float | None, dt_profile: float,
           dz: float = 8.0, smoofac: float = 0.0, botfac: float = 1.0, barofac: float = 1.0,
           velerr_override: float | None = None,
           reject: np.ndarray | None = None) -> _Solution:
    """One weighted L2 inverse solve (getinv body through ``lainsolv``)."""
    nbin, nt = se.ru.shape
    d = (se.ru + 1j * se.rv).reshape(-1)
    izv = (-se.izm).reshape(-1)                              # positive depth per measurement
    jprof = np.repeat(np.arange(nt)[None, :], nbin, axis=0).reshape(-1)
    ibin_full = np.repeat(np.arange(nbin), nt)

    # velerr: median over super-ensembles of the across-bin scatter of the near-transducer
    # vertical velocity, scaled by tan(beam angle) (getinv super-ensemble velocity error).
    nmax = min(se.izd.size, 7)
    sw = np.nanstd(se.rw[se.izd[:nmax]], axis=0)
    sw = sw[sw > 0]
    velerr = max(float(np.nanmedian(sw) / np.tan(np.deg2rad(BEAM_ANGLE))), 0.02) if sw.size else 0.02
    if velerr_override is not None and np.isfinite(velerr_override) and velerr_override > 0:
        velerr = float(velerr_override)
    wm = (velerr / se.ruvs).reshape(-1)

    # getinv zeroes the surface/below-bottom data first, then sizes the depth grid from the
    # *remaining* measurements (its izv>dz*(jmax-1) term is a no-op — jmax is a depth value).
    jz_all = np.round(np.where(np.isfinite(izv), izv, 0) / dz).astype(int)
    zb = (zbottom - dz) if zbottom else 1e30
    good = (np.isfinite(d) & np.isfinite(wm) & np.isfinite(izv)
            & (wm >= WEIGHTMIN) & (izv > dz) & (izv < zb))
    if reject is not None:
        good &= ~reject.reshape(-1)
    d, wm, jz, jprof, ibin = d[good], wm[good], jz_all[good], jprof[good], ibin_full[good]
    nz = int(jz.max())                                       # deepest occupied ocean bin
    z = np.arange(1, nz + 1) * dz
    jz = np.clip(jz, 1, nz) - 1                              # 0-based ocean bin
    ndata = d.size

    rows = np.arange(ndata)
    A2 = sp.coo_matrix((wm, (rows, jz)), shape=(ndata, nz)).tocsr()       # ocean
    A1 = sp.coo_matrix((wm, (rows, jprof)), shape=(ndata, nt)).tocsr()    # package
    rhs = d * wm
    range_ocean = np.asarray(np.abs(A2).sum(axis=0)).ravel()
    range_ctd = np.asarray(np.abs(A1).sum(axis=0)).ravel()

    blocks_o, blocks_c, rhs_parts = [A2], [A1], [rhs]

    def add_block(o_block, c_block, r_vals):
        blocks_o.append(o_block)
        blocks_c.append(c_block)
        rhs_parts.append(np.asarray(r_vals, dtype=complex))

    def smoo_block(rows_spec, ncol):
        nr = len(rows_spec)
        ii = np.concatenate([np.full(c.size, k) for k, (c, _) in enumerate(rows_spec)])
        jj = np.concatenate([c for c, _ in rows_spec])
        vv = np.concatenate([v for _, v in rows_spec])
        return sp.coo_matrix((vv, (ii, jj)), shape=(nr, ncol)).tocsr(), nr

    srows = _lainsmoo(range_ocean, smoofac)
    if srows:
        blk, nr = smoo_block(srows, nz)
        add_block(blk, sp.csr_matrix((nr, nt)), np.zeros(nr))
    crows = _lainsmoo(range_ctd, smoofac)
    if crows:
        blk, nr = smoo_block(crows, nt)
        add_block(sp.csr_matrix((nr, nz)), blk, np.zeros(nr))

    # --- bottom-track constraint (lainbott) ----------------------------------- #
    n_bt = 0
    if zbottom is not None and np.isfinite(zbottom) and botfac > 0 and se.bvel.shape[1] == nt:
        bcomplex = se.bvel[0] + 1j * se.bvel[1]
        bvels_h = np.hypot(se.bvels[0], se.bvels[1])
        wref = np.nanmedian(se.rw[se.izr], axis=0)
        hbot = zbottom + se.z                                # height above bottom [m]
        usable = (np.isfinite(bcomplex) & np.isfinite(bvels_h) & (bvels_h > 0)
                  & (hbot > BTRK_RANGE[0]) & (hbot < BTRK_RANGE[1])
                  & (np.abs(se.bvel[2] - wref) < BTRK_WLIM))
        if usable.any():
            wstd = 2 * float(np.nanmedian(bvels_h[usable]))
            usable &= bvels_h <= wstd
        idx = np.where(usable)[0]
        if idx.size:
            btweight = velerr / bvels_h[idx]
            fac = btweight * botfac * np.sqrt(range_ctd[idx])
            n_bt = idx.size
            cb = sp.coo_matrix((fac, (np.arange(n_bt), idx)), shape=(n_bt, nt)).tocsr()
            add_block(sp.csr_matrix((n_bt, nz)), cb, bcomplex[idx] * fac)

    # --- barotropic (GPS) constraint (lainbaro) ------------------------------- #
    have_baro = barofac > 0 and np.isfinite(uship) and dt_profile > 0
    if have_baro:
        barvelerr = 2 * NAV_ERROR / dt_profile
        w = barofac * (velerr / barvelerr)
        dt = se.dtiv.astype(float)
        gaps = dt > 3 * np.nanmean(dt)
        if gaps.sum() > 1:
            facgap = np.nansum(dt[gaps]) / np.nansum(dt)
            w *= (1 - np.tanh(facgap / 0.15))
        if not np.isfinite(w):
            w = 1.0
        fac = np.sqrt(np.nansum(range_ctd))
        crow = (dt / np.nansum(dt) * w * fac)[None, :]
        add_block(sp.csr_matrix((1, nz)), sp.csr_matrix(crow), [-uship * w * fac])

    if n_bt == 0 and not have_baro:                          # zero-mean fallback
        fac = float(np.mean(range_ocean))
        add_block(sp.csr_matrix(np.ones((1, nz)) * fac), sp.csr_matrix((1, nt)), [0.0])

    Ao = sp.vstack(blocks_o).tocsr()
    Ac = sp.vstack(blocks_c).tocsr()
    rhs_full = np.concatenate(rhs_parts)
    G = sp.hstack([Ao, Ac]).tocsr()
    m_re = lsqr(G, np.real(rhs_full), atol=1e-9, btol=1e-9, iter_lim=8000)[0]
    m_im = lsqr(G, np.imag(rhs_full), atol=1e-9, btol=1e-9, iter_lim=8000)[0]
    uocean = m_re[:nz] + 1j * m_im[:nz]
    uctd = m_re[nz:nz + nt] + 1j * m_im[nz:nz + nt]
    nvel = np.asarray(A2.astype(bool).sum(axis=0)).ravel()
    return _Solution(uocean, uctd, velerr, z, nz, nt, n_bt, nvel, d, jz, jprof, ibin)


def _ocean_error(sol: _Solution) -> np.ndarray:
    """Per-bin velocity error = scatter of (measured - package) across measurements (geterr)."""
    ocean_est = sol.d - sol.uctd[sol.iens]
    uerr = np.full(sol.nz, np.nan)
    for k in range(sol.nz):
        m = sol.jz == k
        if m.sum() >= 2:
            uerr[k] = np.hypot(np.std(np.real(ocean_est[m])), np.std(np.imag(ocean_est[m])))
    return uerr


def _lanarrow_reject(sol: _Solution, shape: tuple[int, int], frac: float = 0.01) -> np.ndarray:
    """Mask of the worst-``frac`` data points by fit residual (legacy ``lanarrow``)."""
    reject = np.zeros(shape, dtype=bool)
    resid = sol.d - (sol.uocean[sol.jz] + sol.uctd[sol.iens])
    dif = np.abs(resid) ** 2
    ok = np.where(np.isfinite(dif))[0]
    n = int(ok.size * frac)
    if ok.size == 0 or n < 1:
        return reject
    worst = ok[np.argsort(dif[ok])[-n:]]
    reject[sol.ibin[worst], sol.iens[worst]] = True
    return reject


def invert(se: SuperEns, *, uship: complex, zbottom: float | None, dt_profile: float,
           dz: float = 8.0, smoofac: float = 0.0, botfac: float = 1.0, barofac: float = 1.0,
           outlier: float = 1.0) -> InverseResult:
    """Two-pass constrained inversion (process_cast STEP 11 + STEP 14).

    Pass 1 uses the super-ensemble-scatter ``velerr``; its residuals drive the 1%
    ``lanarrow`` rejection and the residual-based two-pass ``velerr=median(uerr)``. Pass 2
    re-solves with that error on the cleaned data. ``outlier<=0`` reverts to a single pass.
    """
    sol = _solve(se, uship=uship, zbottom=zbottom, dt_profile=dt_profile,
                 dz=dz, smoofac=smoofac, botfac=botfac, barofac=barofac)
    n_outlier = int(np.ceil(outlier)) if outlier and outlier > 0 else 0
    reject = None
    if n_outlier > 0:
        velerr2 = float(np.nanmedian(_ocean_error(sol)))
        for _ in range(n_outlier):
            new = _lanarrow_reject(sol, se.ru.shape)
            reject = new if reject is None else (reject | new)
            sol = _solve(se, uship=uship, zbottom=zbottom, dt_profile=dt_profile,
                         dz=dz, smoofac=smoofac, botfac=botfac, barofac=barofac,
                         velerr_override=velerr2, reject=reject)

    uerr = _ocean_error(sol)
    uerr = np.where(np.isfinite(uerr), uerr, sol.velerr)
    return InverseResult(
        z=sol.z, u=np.real(sol.uocean), v=np.imag(sol.uocean), uerr=uerr, nvel=sol.nvel,
        ubar=float(np.nanmean(np.real(sol.uocean))), vbar=float(np.nanmean(np.imag(sol.uocean))),
        velerr=sol.velerr, uctd=sol.uctd, n_bt=sol.n_bt,
        config={"dz": dz, "nz": sol.nz, "n_superens": sol.nt, "n_bottom_track": sol.n_bt,
                "smoofac": smoofac, "botfac": botfac, "barofac": barofac,
                "n_outlier_passes": n_outlier},
    )
