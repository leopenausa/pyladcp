"""Phase 5c — shear-method velocity profile (``getshear2.m``).

The shear method is the robust core of the LADCP solution: it never forms the full
least-squares inverse, so it is insensitive to the editing-weight NaN pattern that we
cannot fully reproduce (the documented fidelity wall). For each super-ensemble cell it
takes the **central-difference vertical shear** ``d(ru)/d(izm)``, pools every cell's shear
estimate into ``dz`` (8 m) depth bins with 2-sigma editing, integrates the binned shear
from the bottom up, and removes the mean -- yielding the *baroclinic* (zero-mean) velocity
profile. The magnetic declination ``drot`` rotates it from magnetic to true east/north.

Validated against the MORIA-80 golden ``dr.u_shear_method`` / ``dr.v_shear_method``:
corr 0.99 (u) / 0.96 (v), RMS ~0.02 m/s -- profile-level agreement.

:func:`velocity_profile` adds the **barotropic (depth-mean) reference** to the baroclinic
shear, reproducing the golden ``ps.shear==1`` inverse (which uses the shear method for the
shape and the inverse only for the reference). Validated vs golden ``dr.u`` / ``dr.v``:
corr 0.98 (u) / 0.97 (v), RMS ~0.03 / 0.015 m/s.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import TYPE_CHECKING

import numpy as np

from ..models import CTDTimeSeries
from .depth import synchronize
from .ingest import DualHead
from .superens import SuperEns, _uvrot, form_superensembles, merge_heads

if TYPE_CHECKING:
    from .bottom import BtrkDiagnostics
    from .inverse_full import ConstraintWeights


@dataclass
class ShearProfile:
    """Baroclinic (zero-mean) velocity profile from the shear method."""

    z: np.ndarray               # [nz] depth grid [m, +down]
    u: np.ndarray               # [nz] east  baroclinic velocity [m/s, true]
    v: np.ndarray               # [nz] north baroclinic velocity [m/s, true]
    w: np.ndarray               # [nz] vertical baroclinic velocity [m/s]
    u_shear: np.ndarray         # [nz] binned east shear [1/s]
    v_shear: np.ndarray         # [nz] binned north shear [1/s]
    n: np.ndarray               # [nz] shear samples per bin


def _diff2(x: np.ndarray) -> np.ndarray:
    """Central 2-step difference down axis 0 (legacy ``diff2``): ``x[2:] - x[:-2]``."""
    return x[2:] - x[:-2]


def shear_method(se: SuperEns, *, dz: float = 8.0, drot: float = 0.0,
                 z: np.ndarray | None = None, weightmin: float = 0.1,
                 stdf: float = 2.0) -> ShearProfile:
    """Compute the shear-method baroclinic velocity profile (``getshear2.m``).

    ``se`` is the super-ensemble product (:func:`~ladcp.qa.superens.form_superensembles`).
    ``drot`` is the magnetic declination [deg] (golden ``p.drot``); the profile is rotated
    magnetic->true by it. ``z`` is the output depth grid [m, +down]; if ``None`` an 8 m
    grid spanning the cast is built. Cells with weight <= ``weightmin`` are dropped, and
    each depth bin is averaged after ``stdf``-sigma editing.
    """
    izm_neg = -se.izm                                   # negative-down (legacy convention)
    w = np.where(se.weight > weightmin, 1.0, np.nan)
    pad = np.full((1, se.ru.shape[1]), np.nan)

    def shear(r: np.ndarray) -> np.ndarray:
        return np.vstack([pad, _diff2(r) / _diff2(izm_neg), pad]) * w

    us, vs, ws = shear(se.ru), shear(se.rv), shear(se.rw)
    depth = (-izm_neg).ravel()                          # +down depth per cell
    U, V, W = us.ravel(), vs.ravel(), ws.ravel()

    if z is None:
        valid = depth[np.isfinite(U + V)]              # deepest cell with usable shear
        zmax = float(np.nanmax(valid)) if valid.size else float(np.nanmax(depth))
        z = np.arange(dz, zmax, dz)
    z = np.asarray(z, dtype=float)

    usm = np.full(z.size, np.nan)
    vsm = np.full(z.size, np.nan)
    wsm = np.full(z.size, np.nan)
    nn = np.zeros(z.size, dtype=int)

    for k, i in enumerate(z):
        sel = (np.abs(depth - i - dz / 2.0) <= dz) & np.isfinite(U + V)
        nn[k] = int(sel.sum())
        if nn[k] <= 2:
            continue
        for arr, store in ((U, usm), (V, vsm), (W, wsm)):
            a = arr[sel]
            good = a[np.isfinite(a)]
            if good.size <= 1:
                continue
            keep = np.abs(a - np.median(good)) < stdf * np.std(good)
            if keep.sum() > 1:
                store[k] = np.mean(a[keep])

    # integrate shear from the bottom up, remove the mean (baroclinic)
    ur = np.flip(np.cumsum(np.flip(np.nan_to_num(usm)))) * dz
    vr = np.flip(np.cumsum(np.flip(np.nan_to_num(vsm)))) * dz
    wr = np.flip(np.cumsum(np.flip(np.nan_to_num(wsm)))) * dz
    ur -= ur.mean()
    vr -= vr.mean()
    wr -= wr.mean()
    ur, vr = _uvrot(ur, vr, drot)                       # magnetic -> true

    return ShearProfile(z=z, u=ur, v=vr, w=wr, u_shear=usm, v_shear=vsm, n=nn)


@dataclass
class VelocityProfile:
    """Absolute ocean velocity profile (baroclinic shear + barotropic reference)."""

    z: np.ndarray               # [nz] depth [m, +down]
    u: np.ndarray               # [nz] east  velocity [m/s, true]
    v: np.ndarray               # [nz] north velocity [m/s, true]
    uerr: np.ndarray            # [nz] velocity uncertainty [m/s]
    nvel: np.ndarray            # [nz] velocity samples per bin
    ubar: float                 # barotropic (depth-mean) east  reference [m/s]
    vbar: float                 # barotropic (depth-mean) north reference [m/s]
    n: np.ndarray               # [nz] shear samples per bin


# max near-surface gap (m) the profile may bridge by filling from the shallowest reliable bin.
# A genuine surface gap (up-looker can't quite reach the surface) is small; a larger gap means
# the upper water column was never sampled (ADCP began mid-descent on a fragmented cast) and
# must stay NaN rather than be padded with a fake constant velocity.
_SURFACE_FILL_MAX = 250.0


def velocity_profile(se: SuperEns, *, dz: float = 8.0, drot: float = 0.0,
                     z: np.ndarray | None = None, uship: float = 0.0,
                     vship: float = 0.0, weightmin: float = 0.1, velerr: float = 0.03,
                     uref: float | None = None, vref: float | None = None
                     ) -> VelocityProfile:
    """Absolute velocity profile from the shear method + barotropic reference.

    The golden inverse uses ``ps.shear==1``: the baroclinic shape comes from
    :func:`shear_method`, and the inverse only sets the depth-mean (barotropic) reference.
    We reproduce that directly. With the baroclinic profile fixed, every super-ensemble
    measurement reads ``se.ru = u_baroclinic(z) + ubar - u_package(s)``, so per
    super-ensemble the residual ``mean_b(se.ru - u_baroclinic)`` equals ``ubar - u_package``.
    The barotropic constraint (weighted-mean package velocity over the cast = ship
    velocity, here ~0 because the package returns near its start) then pins ``ubar``.

    ``uref``/``vref`` override that constraint with a precomputed magnetic-frame reference
    -- :func:`compute_velocity` passes the bottom-track reference (:func:`_btrk_reference`,
    legacy ``lainbott``), the more accurate constraint: ``ubar`` -0.069 vs golden -0.065
    (package-return gives -0.041).

    Validated vs MORIA-80 golden ``dr.u`` / ``dr.v``: corr 0.998 (u) / 0.98 (v); with the
    bottom-track reference the depth-mean u-offset drops from ~0.024 to ~0.005 m/s.
    """
    sp = shear_method(se, dz=dz, drot=0.0, z=z, weightmin=weightmin)   # magnetic frame
    ubc, vbc, zg = sp.u, sp.v, sp.z

    depth = se.izm
    ubc_c = np.interp(depth.ravel(), zg, ubc).reshape(depth.shape)
    vbc_c = np.interp(depth.ravel(), zg, vbc).reshape(depth.shape)
    wt = np.where(se.weight > weightmin, se.weight, np.nan)

    def per_se(res: np.ndarray) -> np.ndarray:
        """Weighted per-super-ensemble mean of the baroclinic-removed residual."""
        msk = np.isfinite(res) & np.isfinite(wt)
        num = np.nansum(np.where(msk, res * wt, 0.0), axis=0)
        den = np.nansum(np.where(msk, wt, 0.0), axis=0)
        return np.where(den > 0, num / den, np.nan)

    dtiv = se.dtiv.astype(float)

    def barotropic(res: np.ndarray, ship: float) -> float:
        pkg = per_se(res)                          # ubar - u_package per super-ensemble
        ok = np.isfinite(pkg)
        return float(ship + np.sum(pkg[ok] * dtiv[ok]) / np.sum(dtiv[ok]))

    if uref is not None:
        ubar, vbar = uref, vref
    else:
        ubar = barotropic(se.ru - ubc_c, uship)
        vbar = barotropic(se.rv - vbc_c, vship)

    u, v = _uvrot(ubc + ubar, vbc + vbar, drot)    # magnetic -> true

    # per-bin sample count and uncertainty (edge-inflated ~ velerr*sqrt(nmed/n))
    dep = depth.ravel()
    valid = (np.isfinite(se.ru) & (se.weight > weightmin)).ravel()
    nvel = np.array([int(((np.abs(dep - zc) <= dz) & valid).sum()) for zc in zg])
    nmed = np.median(nvel[nvel > 0]) if np.any(nvel > 0) else 1.0
    uerr = velerr * np.sqrt(nmed / np.clip(nvel, 1, None))
    uerr = np.where(nvel > 0, np.clip(uerr, velerr * 0.6, None), np.nan)

    # fill only a genuine near-SURFACE gap (no data above the shallowest bin) with the nearest
    # reliable value, as the golden .lad does. A gap larger than _SURFACE_FILL_MAX means the
    # upper water column was never sampled (e.g. an ADCP file that began mid-descent on a
    # fragmented cast) -- leave it NaN rather than fabricate a flat profile. The seabed is NOT
    # filled: the down-looker samples to the bottom, so a flat near-bottom tail would be wrong.
    good = nvel >= max(2.0, 0.15 * nmed)
    if good.any():
        lo = int(np.argmax(good))
        if zg[lo] <= _SURFACE_FILL_MAX:
            for a in (u, v, uerr):
                a[:lo] = a[lo]
        else:                                  # large unsampled surface gap -> honest NaN
            samp = np.where(nvel > 0)[0]
            cut = int(samp[0]) if samp.size else lo
            for a in (u, v, uerr):
                a[:cut] = np.nan

    return VelocityProfile(z=zg, u=u, v=v, uerr=uerr, nvel=nvel,
                           ubar=float(np.nanmean(u)), vbar=float(np.nanmean(v)), n=sp.n)


@dataclass
class BottomProfile:
    """Bottom-track-referenced velocity profile (legacy ``dr.zbot``/``ubot``, ``.bot``)."""

    z: np.ndarray               # [nz] depth [m, +down]
    u: np.ndarray               # [nz] east  velocity [m/s, true]
    v: np.ndarray               # [nz] north velocity [m/s, true]
    uerr: np.ndarray            # [nz] uncertainty [m/s]
    n: np.ndarray               # [nz] cell estimates per bin

    @property
    def n_bins(self) -> int:
        return int(np.isfinite(self.u).sum())


def _btrk_cells(merged, bt, sync, *, zbottom: float, weightmin: float = 0.1):
    """Flatten bottom-track cells -> ``(depth, resid_u, resid_v, weight)`` [magnetic].

    Every water cell of a bottom-track ping gives an absolute ocean velocity
    ``ru - re(bvel)`` -- the seabed cell measures ``bvel = -u_package`` and ``ru`` is the
    *raw* merged earth-frame velocity (not the super-ensemble's reference-removed ``se.ru``,
    which cancels the package term). Below-seabed, above-surface and low-weight cells are dropped.
    """
    n = merged.ru.shape[1]
    z_on = np.asarray(sync.z_on_ping[:n], float)
    izm = z_on[None, :] + merged.offset[:, None]
    resid_u = merged.ru - np.real(bt.bvel)[None, :]
    resid_v = merged.rv - np.imag(bt.bvel)[None, :]
    mask = (np.isfinite(bt.bvel)[None, :] & np.isfinite(resid_u) & np.isfinite(resid_v)
            & (merged.weight > weightmin)
            & (izm >= 0.0))      # drop above-surface up-looker cells (shallow-cast "atmosphere"
                                 # bins: izm = package_depth + offset, up-looker offset<0). No-op
                                 # on deep casts (BT pings only fire when the package is deep).
    if np.isfinite(zbottom):
        # pure legacy side-lobe wedge (no flat floor); cellfac 1.5 == 0.015*cell_cm in metres.
        d2r = np.pi / 180.0
        wedge = ((1.0 - np.cos(merged.beam_dn * d2r)) * (zbottom - izm)
                 + 1.5 * merged.cell_dn)
        mask &= izm <= (zbottom - wedge)
    return izm[mask], resid_u[mask], resid_v[mask], merged.weight[mask]


def _btrk_reference(bp_mag: BottomProfile | None,
                    sp: ShearProfile) -> tuple[float, float] | None:
    """Barotropic reference [m/s, magnetic] from the bottom track (legacy ``lainbott``).

    The magnetic bottom-referenced profile ``bp_mag`` is the absolute ocean velocity in the
    overlap band; subtracting the fixed magnetic baroclinic shape ``sp`` (``drot=0``) leaves
    the depth-mean offset. Averaging *per depth bin* (not per cell, which over-weights the
    cell-dense up-looker band) pins the reference: ``ubar`` -0.069 vs golden -0.065. ``None``
    if no bottom track.
    """
    if bp_mag is None:
        return None
    keep = np.isfinite(bp_mag.u)
    if not keep.any():
        return None
    du = bp_mag.u[keep] - np.interp(bp_mag.z[keep], sp.z, sp.u)
    dv = bp_mag.v[keep] - np.interp(bp_mag.z[keep], sp.z, sp.v)
    return float(np.mean(du)), float(np.mean(dv))


def bottom_referenced_profile(merged, bt, sync, *, zbottom: float, dz: float = 8.0,
                              drot: float = 0.0, z: np.ndarray | None = None,
                              weightmin: float = 0.1, velerr: float = 0.03,
                              nmin: int = 3) -> BottomProfile | None:
    """Bottom-track-referenced velocity profile (legacy ``lainbott`` profile branch).

    The bottom-track cells (:func:`_btrk_cells`) carry absolute ocean velocity; they are
    depth-binned on the ``dz`` grid (weighted mean, >= ``nmin`` estimates) and rotated
    magnetic->true by ``drot`` -- the ``.bot`` deliverable. Returns ``None`` if there is no
    bottom track. Validated vs MORIA-80 golden ``dr.ubot``: corr 0.99, RMS ~0.025 m/s.
    """
    depth, resid_u, resid_v, w_all = _btrk_cells(merged, bt, sync, zbottom=zbottom,
                                                 weightmin=weightmin)
    fin = np.isfinite(depth)
    if not fin.any():                    # no seabed (zbottom NaN) -> no bottom-referenced profile
        return None
    if not fin.all():                    # drop non-finite cells (degenerate near-touch casts)
        depth, resid_u, resid_v, w_all = depth[fin], resid_u[fin], resid_v[fin], w_all[fin]
    if z is None:
        z = np.arange(np.floor(depth.min() / dz) * dz, depth.max() + dz, dz)
    z = np.asarray(z, dtype=float)

    u = np.full(z.size, np.nan); v = np.full(z.size, np.nan)
    uerr = np.full(z.size, np.nan); n = np.zeros(z.size, dtype=int)
    for k, zc in enumerate(z):
        sel = np.abs(depth - zc) <= dz / 2.0 + 1e-6
        nn = int(sel.sum())
        if nn < nmin:
            continue
        w = w_all[sel]
        u[k] = float(np.sum(resid_u[sel] * w) / np.sum(w))
        v[k] = float(np.sum(resid_v[sel] * w) / np.sum(w))
        uerr[k] = velerr / np.sqrt(nn)
        n[k] = nn

    keep = np.isfinite(u)
    u[keep], v[keep] = _uvrot(u[keep], v[keep], drot)     # magnetic -> true
    return BottomProfile(z=z, u=u, v=v, uerr=uerr, n=n)


def _build(dh: DualHead, ctd: CTDTimeSeries, *, dz: float, params):
    """Shared front end: sync + bottom detect + filtered bottom track + super-ensembles."""
    from .bottom import bottom_track_velocity, detect_bottom

    sync = synchronize(dh, ctd)
    bottom = detect_bottom(dh, sync, ctd=ctd)
    merged = merge_heads(dh, params=params)
    bt = bottom_track_velocity(dh, merged)
    # prepinv STEP 10: keep bottom track only where the package is 50-300 m above the
    # known seabed and the echo distance agrees with that geometry (<100 m) -- this drops
    # mid-water false echoes that pass the per-ping target-strength test.
    if np.isfinite(bottom.zbottom):
        hgt = bottom.zbottom - np.asarray(sync.z_on_ping[:bt.bvel.size], float)
        # legacy prepinv.m:71-76: the 50-300 m range gate drops bottom-track VELOCITY; the
        # |hgt-hbot|>100 m echo disagreement NaNs only the (inert) echo distance, not bvel.
        bad_vel = ~((hgt > 50.0) & (hgt < 300.0))
        bt.bvel[bad_vel] = np.nan
        bt.bw[bad_vel] = np.nan
        bt.hbot[np.abs(hgt - bt.hbot) >= 100.0] = np.nan
    # merge_heads trims to min(down, up) pings; align z_on_ping (built on the master ping
    # series) to that joint length so super-ensembles see matching arrays when the two heads
    # logged unequal counts (e.g. the fragmented MORIA-01..04 casts). No-op when equal.
    zop = sync.z_on_ping[:merged.ru.shape[1]]
    se = form_superensembles(
        merged, zop, avdz=dz, zbottom=bottom.zbottom,
        dzbelow=getattr(params, "dzbelow", 16.0) if params is not None else 16.0,
        edit_sidelobes=getattr(params, "edit_sidelobes", True) if params is not None else True,
        mask_dn_bins=getattr(params, "edit_mask_dn_bins", (1,)) if params is not None else (1,),
        mask_up_bins=getattr(params, "edit_mask_up_bins", (1,)) if params is not None else (1,))
    zmax = sync.maxdepth if np.isfinite(sync.maxdepth) else float(np.nanmax(se.izm))
    return se, np.arange(dz, zmax, dz), merged, bt, sync, bottom


@dataclass
class ErrField:
    """Per-cell solution-misfit field over (instrument bin, super-ensemble) -- Figure 3.

    Each super-ensemble (column) holds the bins at its package depth, so plotting depth vs
    super-ensemble traces the descent/ascent "V". ``resid_*`` is the per-cell residual about
    the shared baroclinic shape (true frame, NaN where edited out); ``u_oce``/``v_oce`` is
    the solution velocity sampled at each cell; ``binno`` is the signed instrument bin number.
    """

    se_index: np.ndarray            # [nbin, nse] super-ensemble index (column broadcast)
    depth: np.ndarray               # [nbin, nse] cell depth [m, +down], NaN where edited
    resid_u: np.ndarray             # [nbin, nse] cell residual, true east [m/s]
    resid_v: np.ndarray             # [nbin, nse] cell residual, true north [m/s]
    u_oce: np.ndarray               # [nbin, nse] solution east velocity at the cell [m/s]
    v_oce: np.ndarray               # [nbin, nse] solution north velocity at the cell [m/s]
    binno: np.ndarray               # [nbin] signed instrument bin number (+down, -up)
    u_std: float
    v_std: float


@dataclass
class VelocityResult:
    """Everything the velocity figures need from one end-to-end solve."""

    vp: VelocityProfile             # absolute u/v profile (.lad)
    bp: BottomProfile | None        # bottom-track-referenced profile (.bot)
    shear: ShearProfile             # baroclinic shear profile (true frame) -- Figure 3
    zbottom: float
    resid_z: np.ndarray             # [k] cell depths with a finite fit residual -- Figure 12
    resid_u: np.ndarray             # [k] cell minus shear-fit (true east) [m/s]
    resid_v: np.ndarray             # [k] cell minus shear-fit (true north) [m/s]
    sadcp: np.ndarray | None = None  # [m,4] ship-ADCP constraint (z,u,v,verr) true -- Fig 9
    btrk: BtrkDiagnostics | None = None  # own-vs-RDI bottom-track check -- Figure 13
    err: ErrField | None = None     # per-cell misfit / velocity field -- Figure 3
    drift: DriftTrack | None = None  # ship + package horizontal tracks -- drift map
    weights: ConstraintWeights | None = None  # inverse constraint weights -- Figure 12

    @property
    def resid_rms(self) -> float:
        r = np.concatenate([self.resid_u, self.resid_v])
        return float(np.sqrt(np.mean(r ** 2))) if r.size else np.nan


def _solution_residuals(se: SuperEns, sp_mag: ShearProfile, *, drot: float,
                        weightmin: float) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Per-cell scatter about the shear fit (data-model misfit, Figure 12).

    For each super-ensemble the baroclinic shape ``sp_mag`` and the per-super-ensemble mean
    (package velocity + reference) are removed; what remains is how well the shared
    baroclinic profile explains each cell. Returned rotated magnetic->true by ``drot``.
    """
    depth = se.izm
    ubc_c = np.interp(depth.ravel(), sp_mag.z, sp_mag.u).reshape(depth.shape)
    vbc_c = np.interp(depth.ravel(), sp_mag.z, sp_mag.v).reshape(depth.shape)
    wt = np.where(se.weight > weightmin, se.weight, np.nan)
    au, av = se.ru - ubc_c, se.rv - vbc_c

    def demean(res: np.ndarray) -> np.ndarray:
        msk = np.isfinite(res) & np.isfinite(wt)
        num = np.nansum(np.where(msk, res * wt, 0.0), axis=0)
        den = np.nansum(np.where(msk, wt, 0.0), axis=0)
        return res - np.where(den > 0, num / den, np.nan)[None, :]

    ru_r, rv_r = demean(au), demean(av)
    mask = (np.isfinite(ru_r) & np.isfinite(rv_r) & (se.weight > weightmin)
            & (depth > 0))                                  # drop above-surface up cells
    du, dv = _uvrot(ru_r[mask], rv_r[mask], drot)
    return depth[mask], du, dv


def _error_field(se: SuperEns, sp_mag: ShearProfile, vp: VelocityProfile, *,
                 drot: float, weightmin: float, offset: np.ndarray,
                 cell_m: float) -> ErrField:
    """Per-cell residual + ocean-velocity field over (bin, super-ensemble) -- Figure 3.

    Same baroclinic-removed, per-super-ensemble-demeaned residual as
    :func:`_solution_residuals`, but kept as full 2-D arrays (NaN where edited out) so it can
    be imaged against depth and super-ensemble number. ``u_oce``/``v_oce`` is the final
    velocity profile sampled at each cell depth; ``binno`` maps each row to a signed
    instrument bin number from the package offset.
    """
    depth = se.izm
    ubc = np.interp(depth.ravel(), sp_mag.z, sp_mag.u).reshape(depth.shape)
    vbc = np.interp(depth.ravel(), sp_mag.z, sp_mag.v).reshape(depth.shape)
    wt = np.where(se.weight > weightmin, se.weight, np.nan)

    def demean(res: np.ndarray) -> np.ndarray:
        msk = np.isfinite(res) & np.isfinite(wt)
        num = np.nansum(np.where(msk, res * wt, 0.0), axis=0)
        den = np.nansum(np.where(msk, wt, 0.0), axis=0)
        return res - np.where(den > 0, num / den, np.nan)[None, :]

    valid = (se.weight > weightmin) & (depth > 0)
    ru_r = np.where(valid, demean(se.ru - ubc), np.nan)
    rv_r = np.where(valid, demean(se.rv - vbc), np.nan)
    du, dv = _uvrot(ru_r, rv_r, drot)

    u_oce = np.where(valid, np.interp(depth.ravel(), vp.z, vp.u).reshape(depth.shape), np.nan)
    v_oce = np.where(valid, np.interp(depth.ravel(), vp.z, vp.v).reshape(depth.shape), np.nan)

    se_index = np.broadcast_to(np.arange(1, depth.shape[1] + 1), depth.shape).astype(float)
    binno = np.round(offset / cell_m).astype(int)

    def robust_std(a: np.ndarray) -> float:
        a = a[np.isfinite(a)]
        if a.size == 0:
            return np.nan
        return float(1.4826 * np.median(np.abs(a - np.median(a))))   # MAD, outlier-resistant

    return ErrField(
        se_index=se_index, depth=np.where(valid, depth, np.nan),
        resid_u=du, resid_v=dv, u_oce=u_oce, v_oce=v_oce, binno=binno,
        u_std=robust_std(du), v_std=robust_std(dv),
    )


@dataclass
class DriftTrack:
    """Ship and dead-reckoned package (CTD) horizontal tracks during the cast -- drift map.

    Both are local east/north metres relative to the deployment start. The ship track is the
    CTD-merged GPS; the package track integrates the per-super-ensemble package velocity over
    ground (``dtiv`` ensembles x ping interval), so the two diverge as the package descends.
    """

    ship_e: np.ndarray
    ship_n: np.ndarray
    ship_sog: np.ndarray            # ship speed over ground [m/s] (colour)
    pkg_e: np.ndarray
    pkg_n: np.ndarray
    i_bottom: int                   # package index at max depth (cast turnaround)


def _drift_track(se: SuperEns, ctd: CTDTimeSeries, vp: VelocityProfile, *,
                 drot: float, ping_dt: float, weightmin: float = 0.1) -> DriftTrack:
    """Build the ship + dead-reckoned package tracks (legacy drift map)."""
    lat0 = float(np.nanmean(ctd.lat))
    m_lat = 111_320.0
    m_lon = 111_320.0 * np.cos(np.radians(lat0))
    ship_e = (ctd.lon - ctd.lon[0]) * m_lon
    ship_n = (ctd.lat - ctd.lat[0]) * m_lat
    t = ctd.time_elapsed_s.astype(float)
    with np.errstate(invalid="ignore", divide="ignore"):
        sog = np.hypot(np.gradient(ship_e, t), np.gradient(ship_n, t))

    # package velocity over ground per super-ensemble: u_pkg = <u_ocean(cell) - measured>
    ru_t, rv_t = _uvrot(se.ru, se.rv, drot)
    depth = se.izm
    uoc = np.interp(depth.ravel(), vp.z, vp.u).reshape(depth.shape)
    voc = np.interp(depth.ravel(), vp.z, vp.v).reshape(depth.shape)
    wt = np.where(se.weight > weightmin, se.weight, np.nan)

    def wmean(a):
        msk = np.isfinite(a) & np.isfinite(wt)
        num = np.nansum(np.where(msk, a * wt, 0.0), axis=0)
        den = np.nansum(np.where(msk, wt, 0.0), axis=0)
        return np.where(den > 0, num / den, np.nan)

    up = wmean(uoc - ru_t)
    vpk = wmean(voc - rv_t)
    dt = se.dtiv.astype(float) * float(ping_dt)
    pkg_e = np.cumsum(np.nan_to_num(up) * dt)
    pkg_n = np.cumsum(np.nan_to_num(vpk) * dt)
    # loop closure: the package is deployed and recovered at the ship, so detrend the raw
    # dead-reckoning bias linearly to land its endpoints on the ship's (what the inverse's
    # barotropic constraint enforces; without it the integral drifts off by tens of metres).
    n = pkg_e.size
    if n > 1:
        ramp = np.arange(n) / (n - 1)
        pkg_e = pkg_e - (pkg_e[-1] - float(ship_e[-1])) * ramp
        pkg_n = pkg_n - (pkg_n[-1] - float(ship_n[-1])) * ramp
    i_bottom = int(np.nanargmax(se.z)) if se.z.size else 0
    return DriftTrack(ship_e=ship_e, ship_n=ship_n, ship_sog=sog,
                      pkg_e=pkg_e, pkg_n=pkg_n, i_bottom=i_bottom)


def compute_velocity(dh: DualHead, ctd: CTDTimeSeries, *, drot: float = 0.0,
                     dz: float = 8.0, params=None, solver: str = "shear"
                     ) -> VelocityProfile:
    """End-to-end velocity profile: merge -> super-ensembles -> velocity solution.

    Convenience wrapper tying the Phase-5 stages together. ``drot`` is the magnetic
    declination [deg] (magnetic->true rotation); pass the cast value (golden
    ``p.drot``) or compute it from position via :func:`ladcp.proc.magdec`. ``solver``
    chooses ``"shear"`` (shear shape + bottom-track reference) or ``"inverse"`` (full
    constrained least-squares) -- see :func:`compute_velocity_full`.
    """
    return compute_velocity_full(dh, ctd, drot=drot, dz=dz, params=params,
                                 solver=solver).vp


def compute_velocity_and_bottom(dh: DualHead, ctd: CTDTimeSeries, *, drot: float = 0.0,
                                dz: float = 8.0, params=None
                                ) -> tuple[VelocityProfile, BottomProfile | None, float]:
    """Back-compat tuple form of :func:`compute_velocity_full` (``.lad``, ``.bot``, zbottom)."""
    r = compute_velocity_full(dh, ctd, drot=drot, dz=dz, params=params)
    return r.vp, r.bp, r.zbottom


def _ping_dt(dh: DualHead) -> float:
    """Median seconds between down-looker pings (super-ensemble time scaling)."""
    t = (dh.down.time - dh.down.time[0]) / np.timedelta64(1, "s")
    dt = float(np.nanmedian(np.diff(t)))
    return dt if np.isfinite(dt) and dt > 0 else 1.0


def compute_velocity_full(dh: DualHead, ctd: CTDTimeSeries, *, drot: float = 0.0,
                          dz: float = 8.0, params=None, weightmin: float = 0.1,
                          solver: str = "inverse", sadcp: np.ndarray | None = None,
                          sadcpfac: float = 3.0, botfac: float = 1.0,
                          barofac: float = 1.0, smoofac: float = 0.0) -> VelocityResult:
    """End-to-end solve returning every product the velocity figures need.

    ``solver`` selects the velocity (``.lad``) solution, naming the legacy ``ps.shear``
    choice explicitly: ``"inverse"`` (default, ``ps.shear==0``) forms the full constrained
    least-squares inverse (:func:`ladcp.qa.inverse_full.invert`) with bottom-track +
    navigation (+ optional ship-ADCP) constraints; ``"shear"`` (``ps.shear==1``) takes the
    shear method's baroclinic shape plus a barotropic reference. The bottom-track ``.bot``
    profile, shear figure and residuals are produced the same way for both.

    ``sadcp`` is an optional ship-ADCP profile ``(z, u, v, verr)`` in the true frame
    (``solver="inverse"`` only); it adds the ``lainsadcp`` ocean constraint with weight
    ``sadcpfac`` (golden default 3). Build it from a raw VmDAS folder with
    :func:`ladcp.io.sadcp_vmdas.extract_profile`; it is carried on the result for Figure 9.

    ``botfac``/``barofac``/``smoofac`` are the legacy ``ps.botfac``/``ps.barofac``/
    ``ps.smoofac`` constraint weights (inverse only): how strongly the bottom track and
    the GPS barotropic row pull the solution, and the curvature-smoothing strength
    (legacy defaults 1/1/0).
    """
    if solver not in ("shear", "inverse"):
        raise ValueError(f"solver must be 'shear' or 'inverse', got {solver!r}")
    # earth-frame guard: refuse beam-coordinate data rather than emit silent garbage velocities
    # (merge_heads reads vel[0..3] as u,v,w,e -- valid only in the earth frame). A beam->earth
    # transform is not yet implemented; this is the single source of truth that blocks it.
    from ..models import CoordFrame
    for label, head in (("down", dh.down), ("up", dh.up)):
        if head is not None and head.coord_frame != CoordFrame.EARTH:
            raise ValueError(
                f"{label}-looker is in {head.coord_frame.value!r} coordinates; the velocity solve "
                "requires earth-frame velocities. A beam->earth transform is not yet implemented "
                "(the QA report flags this as a coord_frame WARN).")
    se, z, merged, bt, sync, bottom = _build(dh, ctd, dz=dz, params=params)
    sp_mag = shear_method(se, dz=dz, drot=0.0, z=z)         # magnetic baroclinic shape
    bp_mag = bottom_referenced_profile(merged, bt, sync, zbottom=bottom.zbottom, dz=dz,
                                       drot=0.0)
    if solver == "inverse":
        from .inverse_full import inverse_inputs, invert
        # weightmin here is the inverse data cut on velerr/scatter (golden ps.weightmin
        # 0.05), a different quantity from the shear path's correlation-weight cut, so the
        # inverse keeps its own legacy default rather than the shear-oriented argument.
        aux = inverse_inputs(se, bt=bt, sync=sync, ctd=ctd, ping_dt=_ping_dt(dh),
                             zbottom=bottom.zbottom)
        inv_diag: dict = {}
        vp = invert(se, aux, dz=dz, drot=drot, zbottom=bottom.zbottom,
                    botfac=botfac, barofac=barofac, smoofac=smoofac,
                    svel=sadcp, sadcpfac=sadcpfac if sadcp is not None else 0.0,
                    diag=inv_diag)
        weights = inv_diag.get("weights")
    else:
        ref = _btrk_reference(bp_mag, sp_mag)
        uref, vref = ref if ref is not None else (None, None)
        vp = velocity_profile(se, dz=dz, drot=drot, z=z, uref=uref, vref=vref,
                              weightmin=weightmin)
        weights = None
    bp = _rotate_bottom(bp_mag, drot) if bp_mag is not None else None
    shear = shear_method(se, dz=dz, drot=drot, z=z)        # true-frame baroclinic (Fig 3)
    rz, ru, rv = _solution_residuals(se, sp_mag, drot=drot, weightmin=weightmin)
    sadcp_used = sadcp if (solver == "inverse" and sadcp is not None) else None
    from .bottom import btrk_diagnostics
    btrk = btrk_diagnostics(bt, dh) if np.isfinite(bottom.zbottom) else None
    err = _error_field(se, sp_mag, vp, drot=drot, weightmin=weightmin,
                       offset=merged.offset, cell_m=dh.down.cell_m)
    drift = _drift_track(se, ctd, vp, drot=drot, ping_dt=_ping_dt(dh))
    return VelocityResult(vp=vp, bp=bp, shear=shear, zbottom=bottom.zbottom,
                          resid_z=rz, resid_u=ru, resid_v=rv, sadcp=sadcp_used,
                          btrk=btrk, err=err, drift=drift, weights=weights)


def _rotate_bottom(bp: BottomProfile, drot: float) -> BottomProfile:
    """Rotate a magnetic-frame :class:`BottomProfile` to true east/north by ``drot``."""
    u, v = bp.u.copy(), bp.v.copy()
    keep = np.isfinite(u)
    u[keep], v[keep] = _uvrot(u[keep], v[keep], drot)
    return BottomProfile(z=bp.z, u=u, v=v, uerr=bp.uerr, n=bp.n)
