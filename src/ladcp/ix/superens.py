"""Super-ensemble formation for the inverse (prepinv.m, STEP 10).

``prepinv`` collapses the thousands of raw ensembles into a few hundred *super-ensembles*:
the cast is walked in depth, ensembles within each ``avdz`` (8 m) depth-bin are grouped, the
package reference velocity (median over near reference bins) is removed, the residual is
averaged per ADCP bin, and the reference mean is added back. Each super-ensemble carries a
velocity-scatter estimate ``ruvs`` (the data error that sets the inverse weights). A two-pass
4σ/3σ outlier rejection (``outlier.m``) over 5-minute blocks discards spurious cells, then
super-ensembles with zero scatter are de-weighted and the scatter is floored at the single-
ping accuracy.

Faithful to ``prepinv.m`` for ``avens==NaN`` (depth-binned) casts with both heads; the
bottom-track averaging is carried for the inverse but the navigation/compass-offset and
sound-speed branches (handled upstream) are out of scope here.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any

import numpy as np

from .depth import bin_depths
from .ingest import LADCPData, _quiet_nan


@dataclass
class SuperEns:
    """Super-ensemble profiles (≈ legacy ``di`` after prepinv).

    Velocity arrays are ``[nbin, nse]``; ``ruvs`` is the per-cell velocity-scatter error,
    ``izm`` the per-cell depth [m, negative down], ``z`` the per-super-ensemble package depth.
    """

    ru: np.ndarray
    rv: np.ndarray
    rw: np.ndarray
    ruvs: np.ndarray
    weight: np.ndarray
    izm: np.ndarray
    z: np.ndarray
    time: np.ndarray
    hbot: np.ndarray
    bvel: np.ndarray                  # [4, nse]
    bvels: np.ndarray                 # [4, nse]
    dtiv: np.ndarray                  # ensembles per super-ensemble
    izd: np.ndarray
    izu: np.ndarray
    izr: np.ndarray
    counts: dict[str, int] = field(default_factory=dict)
    meta: dict[str, Any] = field(default_factory=dict)

    @property
    def n_se(self) -> int:
        return self.z.size


def _rms(x: np.ndarray) -> float:
    """Root-mean-square over finite values, mean NOT removed (legacy ``rms``)."""
    g = np.isfinite(x)
    if not g.any():
        return np.nan
    return float(np.sqrt(np.sum(x[g] ** 2) / g.sum()))


def _outlier(ru, rv, rw, weight, izd, izu, *, nblock: int, nfac=(4.0, 3.0)) -> dict:
    """Two-pass anomaly rejection over time blocks (``outlier.m``) on super-ensembles.

    For each block and pass, cells whose |anomaly| (value minus per-block bin-median removed)
    exceeds ``nfac*rms`` of the block are NaN'd in ``ru/rv/rw`` and ``weight``. Returns the
    discarded-cell counts for the down and up blocks (the golden log's
    *Outlier discarded N bins down/up looking*).
    """
    counts = {}
    for name, idx in (("outlier_down", izd), ("outlier_up", izu)):
        if idx.size == 0:
            counts[name] = 0
            continue
        u = ru[idx].copy()
        v = rv[idx].copy()
        w = rw[idx].copy()
        dummy = np.zeros_like(w)                       # NaN marks a discard
        nse = w.shape[1]
        sn = int(np.ceil(nse / nblock))
        with _quiet_nan():
            for n in nfac:
                u = u - np.nanmedian(u, axis=0)
                v = v - np.nanmedian(v, axis=0)
                w = w - np.nanmedian(w, axis=0)
                for m in range(sn):
                    ind = np.arange(m * nblock, min((m + 1) * nblock, nse))
                    if ind.size == 0:
                        continue
                    for anom in (w[:, ind], u[:, ind], v[:, ind]):
                        bad = np.abs(anom) > n * _rms(anom)
                        block = dummy[:, ind]
                        block[bad] = np.nan
                        dummy[:, ind] = block
                # apply accumulated mask back into anomaly fields for next pass
                u = u + dummy
                v = v + dummy
                w = w + dummy
        bad = np.isnan(dummy)
        for arr in (ru, rv, rw, weight):
            sub = arr[idx]
            sub[bad] = np.nan
            arr[idx] = sub
        counts[name] = int(bad.sum())
    return counts


def form_superensembles(
    d: LADCPData,
    z: np.ndarray,
    *,
    zbottom: float = np.nan,
    avdz: float = 8.0,
    oversample: float = 1.0,
    avpercent: float = 100.0,
    superens_std_min: float = 0.0731,
    outlier_n: int | None = None,
    outlier_fac: tuple[float, float] = (4.0, 3.0),
) -> SuperEns:
    """Form super-ensembles from edited raw ensembles (``prepinv.m`` STEP 10).

    ``z`` is the package depth [m, negative down] on ping times. Groups ensembles by ``avdz``
    depth movement, removes the reference velocity (median over bins 2-3 of each head),
    averages per bin, and estimates the scatter ``ruvs``. Then runs the two-pass outlier
    rejection, drops empty super-ensembles, de-weights zero-scatter cells, and floors
    ``ruvs`` at ``superens_std_min``. Returns a :class:`SuperEns` with the log counts in
    ``.counts`` (``outlier_down/up``, ``weight_nan_zero_std``, ``ruvs_floored``, ``reduced_len``).
    """
    izm = bin_depths(d, z)
    nbin, nens = d.ru.shape
    # reference bins: down bins 2,3 + up bins 2nd,3rd nearest (prepinv default izr)
    izr = []
    if d.izd.size > 2:
        izr += [d.izd[1], d.izd[2]]
    if d.izu.size > 2:
        izr += [d.izu[1], d.izu[2]]
    izr = np.array(izr, dtype=int)

    w2 = np.where(np.isfinite(d.weight), 1.0, np.nan)        # weight-finite mask
    ru_m, rv_m, rw_m = d.ru * w2, d.rv * w2, d.rw * w2
    izm0 = izm[0, :]                                          # depth of row 0 per ensemble

    groups: list[np.ndarray] = []
    ilast = 1                                                 # 1-indexed (matches MATLAB)
    il = nens
    while ilast < il:
        seg = np.abs(izm0[ilast:il] - izm0[ilast - 1])       # ensembles ilast+1..il
        cross = np.where(seg > avdz)[0]
        k = (cross[0] + 1) if cross.size else (il - ilast)
        i1 = ilast + np.arange(1, k + 1)                     # 1-indexed
        i1l = len(i1) / 2.0 * oversample
        # MATLAB round() is half-away-from-zero; numpy round is half-to-even, which can
        # collapse the final group onto ilast and stall the loop. floor(x+0.5) matches.
        i1 = np.floor(np.mean(i1) + np.arange(-i1l, i1l + 1e-9) + 0.5).astype(int)
        i1 = i1[(i1 >= 1) & (i1 <= il)]
        if i1.size == 1:
            i1 = np.array([i1[0], i1[0]])
        ilast = int(i1.max())
        groups.append(i1 - 1)                                # 0-indexed ensemble indices

    nse = len(groups)
    ru = np.full((nbin, nse), np.nan)
    rv = np.full((nbin, nse), np.nan)
    rw = np.full((nbin, nse), np.nan)
    ruvs = np.full((nbin, nse), np.nan)
    weight = np.full((nbin, nse), np.nan)
    izm_se = np.full((nbin, nse), np.nan)
    z_se = np.full(nse, np.nan)
    t_int = d.time.astype("datetime64[ns]").astype(np.int64)
    t_se = np.zeros(nse, dtype=np.int64)
    hbot = np.full(nse, np.nan)
    bvel = np.full((4, nse), np.nan)
    bvels = np.full((4, nse), np.nan)
    dtiv = np.zeros(nse, dtype=int)

    with _quiet_nan():
        for im, i1 in enumerate(groups):
            ref_u = np.nanmedian(ru_m[np.ix_(izr, i1)], axis=0)   # [n_i1]
            ref_v = np.nanmedian(rv_m[np.ix_(izr, i1)], axis=0)
            ref_w = np.nanmedian(rw_m[np.ix_(izr, i1)], axis=0)
            for ref, src, out in ((ref_u, ru_m, ru), (ref_v, rv_m, rv), (ref_w, rw_m, rw)):
                av = np.nanmean(ref)
                r = np.where(np.isfinite(ref), ref, 0.0)
                out[:, im] = np.nanmean(src[:, i1] - r[None, :], axis=1) + av
            rus = np.nanstd(ru_m[:, i1], axis=1)
            rvs = np.nanstd(rv_m[:, i1], axis=1)
            ruvs[:, im] = np.sqrt(rus ** 2 + rvs ** 2)
            weight[:, im] = np.nanmean(np.where(np.isfinite(d.weight[:, i1]),
                                                d.weight[:, i1], np.nan), axis=1)
            izm_se[:, im] = np.nanmean(izm[:, i1], axis=1)
            z_se[im] = np.nanmean(z[i1])
            t_se[im] = int(np.mean(t_int[i1]))
            hbot[im] = np.nanmean(d.hbot[i1])
            bvel[:, im] = np.nanmean(d.bvel[i1, :], axis=0)
            bv = d.bvel[i1, :].copy()
            bv[:, 2] = bv[:, 2] - ref_w                       # remove mean W before BT std
            bvels[:, im] = np.nanstd(bv, axis=0)
            dtiv[im] = i1.size

    # --- two-pass outlier rejection on super-ensembles ----------------------- #
    if outlier_n is None:
        dtmin = np.mean(np.diff(t_se)) / 1e9 / 60.0 if nse > 1 else 1.0
        outlier_n = int(np.ceil(5.0 / dtmin)) if dtmin > 0 else nse
    counts = _outlier(ru, rv, rw, weight, d.izd, d.izu,
                      nblock=outlier_n, nfac=outlier_fac)

    # --- drop empty super-ensembles ------------------------------------------ #
    with _quiet_nan():
        keep = np.isfinite(np.nanmax(np.where(np.isfinite(ru), ru, np.nan), axis=0))
    sel = np.where(keep)[0]
    ru, rv, rw = ru[:, sel], rv[:, sel], rw[:, sel]
    ruvs, weight, izm_se = ruvs[:, sel], weight[:, sel], izm_se[:, sel]
    z_se, t_se, hbot = z_se[sel], t_se[sel], hbot[sel]
    bvel, bvels, dtiv = bvel[:, sel], bvels[:, sel], dtiv[sel]
    counts["non_finite_removed"] = int(nse - sel.size)

    # --- de-weight zero-scatter cells, floor scatter at single-ping accuracy -- #
    ruvs = ruvs + weight * 0.0
    zero = ruvs == 0
    weight[zero] = np.nan
    counts["weight_nan_zero_std"] = int(zero.sum())
    ruvs = ruvs + weight * 0.0
    low = ruvs < superens_std_min
    ruvs[low] = superens_std_min
    counts["ruvs_floored"] = int(np.count_nonzero(low))
    counts["reduced_len"] = int(z_se.size)

    return SuperEns(
        ru=ru, rv=rv, rw=rw, ruvs=ruvs, weight=weight, izm=izm_se, z=z_se,
        time=t_se.astype("datetime64[ns]"), hbot=hbot, bvel=bvel, bvels=bvels,
        dtiv=dtiv, izd=d.izd, izu=d.izu, izr=izr, counts=counts,
        meta={"avdz": avdz, "oversample": oversample, "outlier_n": outlier_n},
    )
