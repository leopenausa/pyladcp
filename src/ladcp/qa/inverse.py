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

import numpy as np

from ..models import CTDTimeSeries
from .depth import synchronize
from .ingest import DualHead
from .superens import SuperEns, _uvrot, form_superensembles, merge_heads


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

    # fill only the sparsely-sampled SURFACE bins (no data above the shallowest bin) with
    # the nearest reliable value, as the golden .lad does. The seabed is NOT filled: the
    # down-looker samples to the bottom, so a flat near-bottom tail would be an artefact.
    good = nvel >= max(2.0, 0.15 * nmed)
    if good.any():
        lo = int(np.argmax(good))
        for a in (u, v, uerr):
            a[:lo] = a[lo]

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


def _btrk_cells(merged, bt, sync, *, zbottom: float, dzbelow: float = 16.0,
                weightmin: float = 0.1):
    """Flatten bottom-track cells -> ``(depth, resid_u, resid_v, weight)`` [magnetic].

    Every water cell of a bottom-track ping gives an absolute ocean velocity
    ``ru - re(bvel)`` -- the seabed cell measures ``bvel = -u_package`` and ``ru`` is the
    *raw* merged earth-frame velocity (not the super-ensemble's reference-removed ``se.ru``,
    which cancels the package term). Below-seabed and low-weight cells are dropped.
    """
    n = merged.ru.shape[1]
    z_on = np.asarray(sync.z_on_ping[:n], float)
    izm = z_on[None, :] + merged.offset[:, None]
    resid_u = merged.ru - np.real(bt.bvel)[None, :]
    resid_v = merged.rv - np.imag(bt.bvel)[None, :]
    mask = (np.isfinite(bt.bvel)[None, :] & np.isfinite(resid_u) & np.isfinite(resid_v)
            & (merged.weight > weightmin))
    if np.isfinite(zbottom):
        mask &= izm <= (zbottom - dzbelow)
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
    if depth.size == 0:
        return None
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
        bad = ~((hgt > 50.0) & (hgt < 300.0) & (np.abs(hgt - bt.hbot) < 100.0))
        bt.bvel[bad] = np.nan
        bt.bw[bad] = np.nan
    se = form_superensembles(merged, sync.z_on_ping, avdz=dz, zbottom=bottom.zbottom)
    zmax = sync.maxdepth if np.isfinite(sync.maxdepth) else float(np.nanmax(se.izm))
    return se, np.arange(dz, zmax, dz), merged, bt, sync, bottom


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


def compute_velocity(dh: DualHead, ctd: CTDTimeSeries, *, drot: float = 0.0,
                     dz: float = 8.0, params=None) -> VelocityProfile:
    """End-to-end velocity profile: merge -> super-ensembles -> shear + bottom-track ref.

    Convenience wrapper tying the Phase-5 stages together. ``drot`` is the magnetic
    declination [deg] (magnetic->true rotation); pass the cast value (golden
    ``p.drot``) or compute it from position via :func:`ladcp.proc.magdec`. The depth-mean
    reference uses the bottom track when available (:func:`_btrk_reference`).
    """
    return compute_velocity_full(dh, ctd, drot=drot, dz=dz, params=params).vp


def compute_velocity_and_bottom(dh: DualHead, ctd: CTDTimeSeries, *, drot: float = 0.0,
                                dz: float = 8.0, params=None
                                ) -> tuple[VelocityProfile, BottomProfile | None, float]:
    """Back-compat tuple form of :func:`compute_velocity_full` (``.lad``, ``.bot``, zbottom)."""
    r = compute_velocity_full(dh, ctd, drot=drot, dz=dz, params=params)
    return r.vp, r.bp, r.zbottom


def compute_velocity_full(dh: DualHead, ctd: CTDTimeSeries, *, drot: float = 0.0,
                          dz: float = 8.0, params=None, weightmin: float = 0.1
                          ) -> VelocityResult:
    """End-to-end solve returning every product the velocity figures need."""
    se, z, merged, bt, sync, bottom = _build(dh, ctd, dz=dz, params=params)
    sp_mag = shear_method(se, dz=dz, drot=0.0, z=z)         # magnetic baroclinic shape
    bp_mag = bottom_referenced_profile(merged, bt, sync, zbottom=bottom.zbottom, dz=dz,
                                       drot=0.0)
    ref = _btrk_reference(bp_mag, sp_mag)
    uref, vref = ref if ref is not None else (None, None)
    vp = velocity_profile(se, dz=dz, drot=drot, z=z, uref=uref, vref=vref,
                          weightmin=weightmin)
    bp = _rotate_bottom(bp_mag, drot) if bp_mag is not None else None
    shear = shear_method(se, dz=dz, drot=drot, z=z)        # true-frame baroclinic (Fig 3)
    rz, ru, rv = _solution_residuals(se, sp_mag, drot=drot, weightmin=weightmin)
    return VelocityResult(vp=vp, bp=bp, shear=shear, zbottom=bottom.zbottom,
                          resid_z=rz, resid_u=ru, resid_v=rv)


def _rotate_bottom(bp: BottomProfile, drot: float) -> BottomProfile:
    """Rotate a magnetic-frame :class:`BottomProfile` to true east/north by ``drot``."""
    u, v = bp.u.copy(), bp.v.copy()
    keep = np.isfinite(u)
    u[keep], v[keep] = _uvrot(u[keep], v[keep], drot)
    return BottomProfile(z=bp.z, u=u, v=v, uerr=bp.uerr, n=bp.n)
