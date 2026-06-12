"""Phase 5a — super-ensemble preparation (``prepinv.m`` STEP 10).

``prepinv`` collapses the thousands of raw ensembles into a few hundred *super-ensembles*
for the inverse. Before averaging it derives the dual-head **compass offset** that rotates
the up-looker into the down-looker frame, two independent ways:

* **velocity / compass** (:func:`compass_offset`, legacy ``compoff``): bins the cast by
  down-heading (36 bins, +/-5 deg), averages each head's unit heading-vector per bin, and
  takes the angle of the binned cross-product. Well-conditioned. MORIA-80: **-59.93 deg**
  (golden ``p.up_dn_comp_off`` **-60.2323**); the ~0.3 deg residual is the exact in-water
  ensemble set / time-matching legacy uses.
* **tilt** (:func:`tilt_offset`, legacy ``checktilt`` + ``fminsearch``): the rotation that
  best aligns the up/down pitch+roll *fluctuations* (mean removed). The objective is very
  flat near its minimum -- golden's -59.5688 is only ~0.1% worse in cost than our -58.26 --
  so this is an ill-conditioned cross-check, not a precise number. Use
  :func:`compass_offset` as the rotation for velocity.

The ensemble grouping (:func:`group_ensembles`) walks the in-water depth trajectory and
starts a new super-ensemble every ``avdz`` (8 m) of vertical travel. MORIA-80: **223 raw
groups** (golden reduced **218** after dropping empty groups + outlier rejection -- the
velocity-merge increment, not yet built here).
"""

from __future__ import annotations

import warnings
from dataclasses import dataclass, field

import numpy as np

from .depth import SyncResult, water_window
from .ingest import DualHead
from .screen import screen

_PG_FIELD = 3            # percent-good field 4 (earth) used for weighting fallback
# Legacy reuses ``p.outlier_n`` (raw pings / 5 min; golden value 289) as the block length at the
# SE-stage ``outlier.m`` call (``prepinv.m:611``). For typical ``nse < 289`` the anomaly RMS is
# computed over ONE global block, not per-12 local blocks -- see ladcp-editing-rootcause-2026-06.
_SE_OUTLIER_NBLOCK = 289


def _medianan_na0(a: np.ndarray) -> np.ndarray:
    """Legacy ``medianan(x)`` (na=0): per column the single sorted order statistic at MATLAB
    ``round(L/2)`` of the finite values (half-away-from-zero). Reproduces ``prepinv.m:524``."""
    a = np.asarray(a, float)
    flat = a.reshape(a.shape[0], -1)
    # Vectorized: np.sort pushes NaN to the end of each column, so the finite values occupy
    # rows [0, L); take the MATLAB round(L/2) order statistic per column in one shot.
    L = np.isfinite(flat).sum(axis=0)                            # [M] finite count per column
    srt = np.sort(flat, axis=0)                                  # NaN -> end of each column
    k = np.clip(np.floor(L / 2 + 0.5).astype(int) - 1, 0, None)  # 0-based order-stat index
    out = np.take_along_axis(srt, k[None, :], axis=0)[0]
    out[L == 0] = np.nan
    return out.reshape(a.shape[1:])


def _uvrot(u: np.ndarray, v: np.ndarray, rot_deg: float) -> tuple[np.ndarray, np.ndarray]:
    """Rotate the (u, v) vector by ``rot_deg`` degrees (legacy ``uvrot.m``)."""
    r = -np.radians(rot_deg)
    cr, sr = np.cos(r), np.sin(r)
    return u * cr - v * sr, u * sr + v * cr


_COMPOFF_DECK_TOL = 5.0      # full-record vs in-water disagreement [deg] -> deck corruption


def compass_offset(dh: DualHead, window: tuple[int, int] | None = None) -> float:
    """Dual-head compass offset [deg] from heading comparison (legacy ``compoff``).

    Bins the cast by down-looker heading (36 bins, +/-5 deg), averages each head's unit
    heading-vector per bin, and returns the angle of the binned cross-product. NaN
    headings fall outside every bin and are excluded automatically.

    ``window`` is the in-water ping span ``(i0, i1)`` when known. The estimate runs
    over the full record (most heading coverage) and is cross-checked against the
    in-water-only estimate: when the two disagree by more than ~5 deg the deck
    segment is corrupting the comparison (the up-looker compass can read garbage on
    deck -- cruise-2 t2-02's full-record estimate was 17 deg off, rotating the merged
    up-looker velocities and putting ~6 cm/s of cross-component error on the cast)
    and the in-water value -- the span the solution actually uses -- wins.
    """
    full = _compass_offset_span(dh, None)
    if not np.isfinite(full):
        full = 0.0                                  # legacy compoff fallback
    if window is None:
        return full
    inw = _compass_offset_span(dh, window)
    if not np.isfinite(inw):
        return full
    diff = abs((inw - full + 180.0) % 360.0 - 180.0)
    return inw if diff > _COMPOFF_DECK_TOL else full


def _compass_offset_span(dh: DualHead, window: tuple[int, int] | None) -> float:
    hd = np.asarray(dh.down.heading, dtype=float)
    hu = np.asarray(dh.up.heading, dtype=float)
    n = min(hd.size, hu.size)
    hd, hu = hd[:n], hu[:n]
    if window is not None:
        i0, i1 = max(0, window[0]), min(n - 1, window[1])
        if i1 - i0 < 10:
            return float("nan")
        hd, hu = hd[i0:i1 + 1], hu[i0:i1 + 1]

    u1 = np.exp(-1j * np.radians(hd))            # down unit heading-vector
    u2 = np.exp(-1j * np.radians(hu))            # up   unit heading-vector
    h1 = -np.angle(u1) * 180 / np.pi             # recovered down heading [0,360)
    h1 = h1 + (h1 < 0) * 360

    nhead = 36
    dhead = 360 / 2 / nhead                      # half bin width = 5 deg
    h0 = np.linspace(5, 355, nhead)
    u1a = np.full(nhead, np.nan, dtype=complex)
    u2a = np.full(nhead, np.nan, dtype=complex)
    for i in range(nhead):
        ii = np.where(np.abs(h1 - h0[i]) <= dhead)[0]
        if ii.size > 1:
            u1a[i] = u1[ii].mean()
            u2a[i] = u2[ii].mean()
    ii = np.where(np.isfinite(u1a + u2a))[0]
    if ii.size == 0:
        return float("nan")
    # MATLAB mrdivide of two row vectors: scalar least-squares u1a/u2a
    num = np.sum(u1a[ii] * np.conj(u2a[ii]))
    den = np.sum(np.abs(u2a[ii]) ** 2)
    return float(np.angle(num / den) * 180 / np.pi)


def tilt_offset(dh: DualHead) -> float:
    """Dual-head offset [deg] that best aligns up/down tilt fluctuations (``checktilt``).

    Minimises the variance of the up-minus-rotated-down pitch and roll (legacy
    ``fminsearch('checktilt', 0, ...)``). The objective is flat near the optimum, so this
    is a cross-check on :func:`compass_offset`, not an independent precise value.
    """
    from scipy.optimize import fmin

    rd = np.asarray(dh.down.roll, dtype=float)
    pd_ = np.asarray(dh.down.pitch, dtype=float)
    ru = np.asarray(dh.up.roll, dtype=float)
    pu = np.asarray(dh.up.pitch, dtype=float)
    n = min(rd.size, ru.size)
    rd, pd_, ru, pu = rd[:n], pd_[:n], ru[:n], pu[:n]

    def cost(rot: np.ndarray) -> float:
        r21, p21 = _uvrot(rd, pd_, float(np.atleast_1d(rot)[0]))
        dr = r21 - ru
        dp = p21 - pu
        return float(np.nansum((dr - np.nanmean(dr)) ** 2 + (dp - np.nanmean(dp)) ** 2))

    return float(fmin(cost, 0.0, disp=False, xtol=1e-4, ftol=1e-6)[0])


def group_ensembles(z: np.ndarray, *, avdz: float = 8.0, oversample: float = 1.0,
                    threshold: float = 10.0) -> list[np.ndarray]:
    """Super-ensemble grouping by depth travel (``prepinv.m`` STEP 10 walk).

    ``z`` is the package depth [m, +down] per ping (NaN out of water). The walk is run on
    the in-water span (internal gaps interpolated so it stays continuous) and a new
    super-ensemble is started every ``avdz`` metres of vertical travel. Returns a list of
    arrays of **original ping indices** (so the velocity merge can index back).
    """
    z = np.asarray(z, dtype=float)
    i0, i1 = water_window(z, threshold)
    span = np.arange(i0, i1 + 1)
    zc = z[i0:i1 + 1]
    ok = np.isfinite(zc)
    if ok.sum() < 2:
        return []
    zc = np.interp(np.arange(zc.size), np.where(ok)[0], zc[ok])

    groups: list[np.ndarray] = []
    ilast = 1                                    # 1-indexed, matches MATLAB
    il = zc.size
    while ilast < il:
        seg = np.abs(zc[ilast:il] - zc[ilast - 1])
        cross = np.where(seg > avdz)[0]
        k = (cross[0] + 1) if cross.size else (il - ilast)
        grp = ilast + np.arange(1, k + 1)
        half = grp.size / 2.0 * oversample
        # MATLAB round() is half-away-from-zero; floor(x+0.5) avoids numpy's
        # half-to-even, which can collapse the final group and stall the loop.
        grp = np.floor(np.mean(grp) + np.arange(-half, half + 1e-9) + 0.5).astype(int)
        grp = grp[(grp >= 1) & (grp <= il)]
        if grp.size == 1:
            grp = np.array([grp[0], grp[0]])
        ilast = int(grp.max())
        groups.append(span[grp - 1])             # back to original ping indices
    return groups


@dataclass
class PrepOffsets:
    """Dual-head offsets + grouping summary from ``prepinv`` STEP 10."""

    compass_deg: float       # velocity/compass offset (use this for the up->down rotation)
    tilt_deg: float          # tilt cross-check (flat objective, indicative)
    n_groups: int            # raw super-ensemble count (before empty-group drop)


def prepinv_offsets(dh: DualHead, sync: SyncResult, *, avdz: float = 8.0) -> PrepOffsets:
    """Compute both dual-head offsets and the raw super-ensemble count for a cast."""
    return PrepOffsets(
        compass_deg=compass_offset(dh),
        tilt_deg=tilt_offset(dh),
        n_groups=len(group_ensembles(sync.z_on_ping, avdz=avdz)),
    )


# --------------------------------------------------------------------------- #
# Dual-head merge + super-ensemble averaging (prepinv.m STEP 10, velocity)
# --------------------------------------------------------------------------- #
def _quiet_nan():
    return warnings.catch_warnings()


@dataclass
class MergedHeads:
    """Both heads stacked on one bin axis, rotated into the down frame (legacy ``d``).

    Velocity arrays are ``[nbin, nens]`` with ``nbin = nbin_up + nbin_down`` in stack
    order ``[flipud(up); down]`` (row 0 = farthest up bin, last row = deepest down bin).
    ``izu``/``izd`` are the combined-row indices of each block, ordered near->far.
    """

    ru: np.ndarray              # east [nbin, nens]
    rv: np.ndarray              # north
    rw: np.ndarray              # vertical
    re: np.ndarray              # error velocity
    weight: np.ndarray          # correlation-based weight [nbin, nens]
    offset: np.ndarray          # [nbin] signed bin distance from package (+down, -up)
    izd: np.ndarray
    izu: np.ndarray
    hrot: np.ndarray            # [nens] per-ping up->down rotation applied [deg]
    std_min: float = np.nan     # super-ensemble scatter floor (Single_Ping_Err/sqrt(ppe))
    beam_dn: float = 20.0       # down-looker beam angle [deg]  (side-lobe geometry)
    beam_up: float = 20.0       # up-looker beam angle [deg]
    cell_dn: float = 0.0        # down-looker cell length [m]
    cell_up: float = 0.0        # up-looker cell length [m]


def merge_heads(dh: DualHead, *, rot_deg: float | None = None,
                params=None, window: tuple[int, int] | None = None) -> MergedHeads:
    """Stack + rotate the two heads into the down-looker frame (loadrdi + prepinv STEP 10).

    Earth-frame data only (no beam transform). The up-looker (u, v) is rotated to the down
    frame with the ``rotup2down==1`` half-difference scheme: per ping, down is rotated by
    ``-hrot/2`` and up by ``+hrot/2`` where ``hrot = angle(u1uc / u1d)`` and
    ``u1uc = exp(-i(hdg_up - hoff))``. ``hoff`` defaults to :func:`compass_offset`.
    Screening edits (:func:`screen`) NaN bad cells; the weight is the beam-median
    correlation normalised by the median of its per-ping maximum.

    With no up-looker (``dh.up is None``) the stack is the down head alone in its own
    compass frame: no rotation (legacy single-head processing), ``izu`` empty, the depth
    offsets purely below the package.
    """
    d, u = dh.down, dh.up
    single = u is None
    hoff = 0.0 if single else (compass_offset(dh, window=window)
                               if rot_deg is None else rot_deg)
    n = d.n_ens if single else min(d.n_ens, u.n_ens)   # joint ensembles (shift 0)
    sr = screen(dh, params)

    def head_fields(head, good, ne):
        vel = head.vel[:, :, :ne].astype(float)
        uu, vv, ww, ee = vel[0], vel[1], vel[2], vel[3]
        with _quiet_nan():
            warnings.simplefilter("ignore", RuntimeWarning)
            corr = np.nanmedian(head.corr[:, :, :ne], axis=0)   # [ncell, ne]
        if good is not None:
            mask = ~good[:, :ne]
            for a in (uu, vv, ww, ee):
                a[mask] = np.nan
            corr = corr.copy()
            corr[mask] = np.nan
        return uu, vv, ww, ee, corr

    ud, vd, wd, ed, cd = head_fields(d, sr.good_down, n)

    if single:
        ru, rv, rw, re = ud, vd, wd, ed                # down head alone, own compass frame
        corr = cd
        hrot = np.zeros(n)
        nbin_u = 0
        offset = dh.bin_depth(d)                       # +down depth offset from package
    else:
        uu, vu, wu, eu, cu = head_fields(u, sr.good_up, n)

        # per-ping rotation hrot (deg), rotup2down==1 (golden)
        hdg_d = np.asarray(d.heading[:n], float)
        hdg_u = np.asarray(u.heading[:n], float)
        u1d = np.exp(-1j * np.radians(hdg_d))
        u1uc = np.exp(-1j * np.radians(hdg_u - hoff))
        hrot = np.angle(u1uc / u1d) * 180 / np.pi
        hrm = np.where(np.isfinite(hrot), hrot, np.nanmean(hrot))
        ud, vd = _uvrot(ud, vd, -hrm / 2.0)            # rotate down by -hrot/2
        uu, vu = _uvrot(uu, vu, hrm / 2.0)             # rotate up   by +hrot/2

        # stack [flipud(up); down]
        def stack(a_up, a_dn):
            return np.vstack([np.flipud(a_up), a_dn])
        ru = stack(uu, ud); rv = stack(vu, vd); rw = stack(wu, wd); re = stack(eu, ed)
        corr = stack(cu, cd)
        nbin_u = u.n_cells
        zu = dh.bin_depth(u)                           # up bin distance from transducer
        offset = np.concatenate([-zu[::-1], dh.bin_depth(d)])   # +down offset from package

    with _quiet_nan():
        warnings.simplefilter("ignore", RuntimeWarning)
        weight = corr / np.nanmedian(np.nanmax(corr, axis=0))

    nbin_d = d.n_cells
    izu = np.flip(np.arange(nbin_u))                   # near->far up rows (empty if single)
    izd = np.arange(nbin_d) + nbin_u                   # near->far down rows

    # single-ping scatter floor from edited down-W, bins 2..6 (loadrdi.m l.207)
    beam = d.meta.get("beam_angle_deg", 20.0)
    ppe = float(d.pings_per_ens or 1)
    with _quiet_nan():
        warnings.simplefilter("ignore", RuntimeWarning)
        nmax = min(nbin_d, 6)
        sw = np.nanstd(rw[izd[1:nmax]], axis=0)
        sw = float(np.nanmedian(sw[sw > 0]))
    std_min = sw / np.tan(np.radians(beam)) / ppe

    beam_up = float(u.meta.get("beam_angle_deg", beam)) if not single else float(beam)
    return MergedHeads(ru=ru, rv=rv, rw=rw, re=re, weight=weight, offset=offset,
                       izd=izd, izu=izu, hrot=hrot, std_min=std_min,
                       beam_dn=float(beam), beam_up=beam_up,
                       cell_dn=float(d.cell_m),
                       cell_up=float(u.cell_m) if not single else float(d.cell_m))


@dataclass
class SuperEns:
    """Super-ensemble profiles after prepinv STEP 10 averaging."""

    ru: np.ndarray              # [nbin, nse] east
    rv: np.ndarray
    rw: np.ndarray
    ruvs: np.ndarray            # per-cell velocity scatter (sets inverse weights)
    weight: np.ndarray
    izm: np.ndarray             # [nbin, nse] per-cell depth [m, +down]
    z: np.ndarray               # [nse] package depth per super-ensemble
    dtiv: np.ndarray            # ensembles per super-ensemble
    izd: np.ndarray
    izu: np.ndarray
    counts: dict[str, int] = field(default_factory=dict)
    group_pings: list[np.ndarray] | None = None   # [nse] original ping indices per kept SE

    @property
    def n_se(self) -> int:
        return self.z.size


def _rms(x: np.ndarray) -> float:
    g = np.isfinite(x)
    return float(np.sqrt(np.sum(x[g] ** 2) / g.sum())) if g.any() else np.nan


def _outlier(ru, rv, rw, weight, izd, izu, *, nblock, nfac=(4.0, 3.0)) -> dict:
    """Two-pass anomaly rejection over time blocks (``outlier.m``)."""
    counts = {}
    for name, idx in (("outlier_down", izd), ("outlier_up", izu)):
        if idx.size == 0:
            counts[name] = 0
            continue
        a_u, a_v, a_w = ru[idx].copy(), rv[idx].copy(), rw[idx].copy()
        dummy = np.zeros_like(a_w)
        nse = a_w.shape[1]
        sn = int(np.ceil(nse / nblock)) if nblock > 0 else 1
        with _quiet_nan():
            warnings.simplefilter("ignore", RuntimeWarning)
            for nf in nfac:
                a_u = a_u - np.nanmedian(a_u, axis=0)
                a_v = a_v - np.nanmedian(a_v, axis=0)
                a_w = a_w - np.nanmedian(a_w, axis=0)
                for m in range(sn):
                    ind = np.arange(m * nblock, min((m + 1) * nblock, nse))
                    if ind.size == 0:
                        continue
                    for anom in (a_w[:, ind], a_u[:, ind], a_v[:, ind]):
                        bad = np.abs(anom) > nf * _rms(anom)
                        blk = dummy[:, ind]; blk[bad] = np.nan; dummy[:, ind] = blk
                a_u = a_u + dummy; a_v = a_v + dummy; a_w = a_w + dummy
        bad = np.isnan(dummy)
        for arr in (ru, rv, rw, weight):
            sub = arr[idx]; sub[bad] = np.nan; arr[idx] = sub
        counts[name] = int(bad.sum())
    return counts


def _edit_velocity_mask(
    izm_full: np.ndarray, z: np.ndarray, merged: MergedHeads, *,
    zbottom: float | None, edit_sidelobes: bool, dzbelow: float,
    mask_dn_bins: tuple[int, ...], mask_up_bins: tuple[int, ...],
    manual_flags: tuple[tuple[str, int, int, int, int], ...] = (),
    manual_counts: dict | None = None,
) -> np.ndarray:
    """Legacy ``edit_data.m`` velocity edits as a boolean ``[nbin, nens]`` removal mask.

    Operates on the merged depth grid (``izm_full = package_depth + bin_offset``, +down) and
    returns the cells to set NaN in the velocity *before* super-ensemble averaging -- the edits
    the goldens ran by default but pyladcp previously applied only to the Figure-14 target-
    strength field:

    * **bin masking** -- the nearest-transducer bin of each head (ringing), rows ``izu[0]`` /
      ``izd[0]`` (legacy ``edit_mask_up_bins`` / ``edit_mask_dn_bins``, golden = bin 1).
    * **manual rectangles** (``manual_flags``, replayed from an edit journal --
      see :mod:`ladcp.edits`): per-head ``(head, bin_first, bin_last, ens_first,
      ens_last)`` cells, bins 1-based inclusive, ensembles 0-based inclusive in the
      joint-trimmed space. Clamped here; out-of-range (or up-head on a single-head
      merge) is a no-op. ``manual_counts``, when given, receives the per-head number
      of finite velocity cells the rectangles *newly* removed.
    * **side-lobe contamination** (``edit_sidelobes``): cells the slanted-beam side lobe reaches
      a hard boundary through -- near the surface (up-looker reflection,
      ``< (1-cos B_up) z + 1.5*cell``) and, *range-dependently*, near the seabed (down-looker,
      ``> zbottom - [(1-cos B_dn)(zbottom - z) + 1.5*cell]``). The seabed cut is the PURE legacy
      wedge (``edit_data.m:147-154``) -- it grows with height above bottom and carries NO flat
      floor; ``1.5*cell`` is legacy's ``0.015*Cell_length`` (cm) expressed in metres. ``dzbelow``
      is used only for the no-side-lobe fallback below.

    The surface side-lobe and bin masking need no seabed and always apply; the seabed removal
    applies only when ``zbottom`` is finite. With ``edit_sidelobes=False`` the seabed cut falls
    back to the flat ``zbottom - dzbelow``.
    """
    nbin, nens = izm_full.shape
    mask = np.zeros((nbin, nens), dtype=bool)
    fin = np.isfinite(izm_full)
    izu, izd = merged.izu, merged.izd

    for b in mask_up_bins:                                  # nearest-TX up bin (izu near->far)
        if 1 <= b <= izu.size:
            mask[izu[b - 1]] = True
    for b in mask_dn_bins:                                  # nearest-TX down bin
        if 1 <= b <= izd.size:
            mask[izd[b - 1]] = True

    # legacy 0.015*Cell_length consumes CENTIMETRES (== 1.5 bins); pyladcp's cell_* is in metres,
    # so the faithful side-lobe margin factor is 1.5 (0.015*cell_cm == 1.5*cell_m).
    cellfac = 1.5
    have_bottom = zbottom is not None and np.isfinite(zbottom)
    if edit_sidelobes:
        d2r = np.pi / 180.0
        sl_surf = (1.0 - np.cos(merged.beam_up * d2r)) * z[None, :] + cellfac * merged.cell_up
        mask |= fin & (izm_full < sl_surf)
    if have_bottom:
        # near-seabed cut: the flat dzbelow floor (preserves the validated deep profile) extended
        # upward by the range-dependent side-lobe wedge (grows with height above bottom).
        margin = float(dzbelow)
        if edit_sidelobes:
            wedge = ((1.0 - np.cos(merged.beam_dn * d2r)) * (zbottom - z[None, :])
                     + cellfac * merged.cell_dn)
            # legacy edit_data seabed cut is the PURE range-dependent wedge -- no flat floor (the
            # only legacy "16" is getdpthi's cut BELOW the bed, a disjoint region).
            sl_bot = zbottom - wedge
        else:
            sl_bot = zbottom - margin
        mask |= fin & (izm_full > sl_bot)

    # manual journal rectangles last, so the per-head "newly removed" count is
    # honest w.r.t. every automatic edit above (the OR is order-independent).
    # Head bin b (1-based) -> merged row izd[b-1]/izu[b-1], the rows ordered
    # near->far exactly like the bin masks; columns are already joint-trimmed.
    for head, b0, b1, e0, e1 in manual_flags:
        rows_idx = merged.izd if head == "down" else merged.izu
        lo, hi = max(int(b0), 1), min(int(b1), rows_idx.size)
        c0, c1 = max(int(e0), 0), min(int(e1), nens - 1)
        if rows_idx.size == 0 or lo > hi or c0 > c1:
            continue                                    # clamped away: silent no-op
        sub = np.ix_(rows_idx[lo - 1:hi], np.arange(c0, c1 + 1))
        if manual_counts is not None:
            newly = ~mask[sub] & np.isfinite(merged.ru[sub])
            key = f"manual_removed_{head}"
            manual_counts[key] = manual_counts.get(key, 0) + int(newly.sum())
        mask[sub] = True
    return mask


def form_superensembles(merged: MergedHeads, z: np.ndarray, *, avdz: float = 8.0,
                        superens_std_min: float | None = None,
                        zbottom: float | None = None, dzbelow: float = 16.0,
                        edit_sidelobes: bool = True,
                        mask_dn_bins: tuple[int, ...] = (1,),
                        mask_up_bins: tuple[int, ...] = (1,),
                        manual_flags: tuple[tuple[str, int, int, int, int], ...] = ()
                        ) -> SuperEns:
    """Average merged ensembles into super-ensembles (``prepinv.m`` STEP 10).

    Groups by depth travel (:func:`group_ensembles`), removes the reference velocity
    (median over near bins of each head), averages per bin, estimates the scatter
    ``ruvs``, runs the two-pass outlier rejection, drops empty super-ensembles and floors
    the scatter at the single-ping accuracy. ``.counts['reduced_len']`` is the golden
    *reduced ensemble size*.

    Before averaging it applies the legacy ``edit_data`` velocity edits
    (:func:`_edit_velocity_mask`): bin masking, side-lobe contamination
    (``edit_sidelobes``) and the below-bottom margin. These remove cells whose slanted-beam
    side lobe reflects off the surface or the seabed -- the down-looker's near-seabed bins
    otherwise bias and flatten the deepest profile, and the wedge is range-dependent, so the
    edit matters most on shallow casts. ``.counts['edit_removed']`` records the cells dropped.
    """
    if superens_std_min is None:
        superens_std_min = merged.std_min
    z = np.asarray(z, float)
    groups = group_ensembles(z, avdz=avdz)
    izd, izu = merged.izd, merged.izu
    nbin = merged.ru.shape[0]
    # reference bins: bins 2-3 of each head (down only when there is no up-looker)
    izr = np.array(([izd[1], izd[2]] + ([izu[1], izu[2]] if izu.size >= 3 else [])),
                   dtype=int)
    izm_full = z[None, :] + merged.offset[:, None]

    manual_counts: dict = {}
    edit = _edit_velocity_mask(izm_full, z, merged, zbottom=zbottom,
                               edit_sidelobes=edit_sidelobes, dzbelow=dzbelow,
                               mask_dn_bins=mask_dn_bins, mask_up_bins=mask_up_bins,
                               manual_flags=manual_flags, manual_counts=manual_counts)
    n_edit = int(edit.sum())
    ru_src = np.where(edit, np.nan, merged.ru)
    rv_src = np.where(edit, np.nan, merged.rv)
    rw_src = np.where(edit, np.nan, merged.rw)

    nse = len(groups)
    ru = np.full((nbin, nse), np.nan); rv = np.full((nbin, nse), np.nan)
    rw = np.full((nbin, nse), np.nan); ruvs = np.full((nbin, nse), np.nan)
    weight = np.full((nbin, nse), np.nan); izm = np.full((nbin, nse), np.nan)
    z_se = np.full(nse, np.nan); dtiv = np.zeros(nse, dtype=int)

    with _quiet_nan():
        warnings.simplefilter("ignore", RuntimeWarning)
        for im, g in enumerate(groups):
            for src, out in ((ru_src, ru), (rv_src, rv), (rw_src, rw)):
                # legacy per-ping reference ur=medianan(d.ru(izr,i1)) (na=0; prepinv.m:524) -- a
                # single sorted order statistic, NOT numpy's mean-of-two-centre median.
                ref = _medianan_na0(src[np.ix_(izr, g)])   # [n_g] per-ping reference
                av = np.nanmean(ref)
                r = np.where(np.isfinite(ref), ref, 0.0)
                out[:, im] = np.nanmean(src[:, g] - r[None, :], axis=1) + av
            rus = np.nanstd(ru_src[:, g], axis=1)
            rvs = np.nanstd(rv_src[:, g], axis=1)
            ruvs[:, im] = np.sqrt(rus ** 2 + rvs ** 2)
            weight[:, im] = np.nanmean(merged.weight[:, g], axis=1)
            izm[:, im] = np.nanmean(izm_full[:, g], axis=1)
            z_se[im] = np.nanmean(z[g])
            dtiv[im] = g.size

    counts = _outlier(ru, rv, rw, weight, izd, izu, nblock=_SE_OUTLIER_NBLOCK)
    counts["edit_removed"] = n_edit
    if manual_flags:                       # keys present only when a journal applied
        counts["manual_removed_down"] = manual_counts.get("manual_removed_down", 0)
        counts["manual_removed_up"] = manual_counts.get("manual_removed_up", 0)

    with _quiet_nan():
        warnings.simplefilter("ignore", RuntimeWarning)
        keep = np.isfinite(np.nanmax(np.where(np.isfinite(ru), ru, np.nan), axis=0))
    sel = np.where(keep)[0]
    counts["non_finite_removed"] = int(nse - sel.size)
    ru, rv, rw = ru[:, sel], rv[:, sel], rw[:, sel]
    ruvs, weight, izm = ruvs[:, sel], weight[:, sel], izm[:, sel]
    z_se, dtiv = z_se[sel], dtiv[sel]

    ruvs = ruvs + weight * 0.0
    zero = ruvs == 0
    weight[zero] = np.nan
    counts["weight_nan_zero_std"] = int(zero.sum())
    low = ruvs < superens_std_min
    ruvs[low] = superens_std_min
    counts["ruvs_floored"] = int(np.count_nonzero(low))
    counts["reduced_len"] = int(z_se.size)

    # surviving ping-index groups (post empty-SE drop), for solvers that need to map
    # per-ping products (bottom-track, nav) back onto the super-ensemble columns.
    group_pings = [groups[i] for i in sel]

    return SuperEns(ru=ru, rv=rv, rw=rw, ruvs=ruvs, weight=weight, izm=izm, z=z_se,
                    dtiv=dtiv, izd=izd, izu=izu, counts=counts, group_pings=group_pings)
