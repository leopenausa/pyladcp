"""Phase 2 — threshold screening (the ``loadrdi`` edits).

Reproduces the per-cell and per-ensemble rejections the legacy ``loadrdi.m`` applies
before any inversion, and counts them so they can be gated against the golden ``.log``:

* percent-good (field 4) < ``pglim``            → cell NaN   (golden 59727 dn / 65340 up)
* error velocity (earth comp 4) > ``elim``       → cell NaN   (golden 4108 combined)
* tilt > ``tiltmax[0]``                          → ensemble down-weighted (golden 6)
* |d tilt| > ``tiltmax[1]``                      → ensemble down-weighted (golden 116)
* horizontal speed > ``vlim``                    → cell NaN   (reported; see note)

Tilt uses the down-looker: ``tilt = asin(sqrt(sin^2 pitch + sin^2 roll))`` and a
centred absolute-difference derivative, exactly as ``loadrdi.m`` lines 407-415.

NOTE on horizontal speed: the legacy count is taken on the *merged* down+up velocity
array (after reference handling), which belongs to the velocity stage. Here it is
computed per head on raw earth u/v, so the combined count is indicative, not bit-exact.
The bit-exact gate stops at pg/errvel/tilt — the edits that are unambiguous on raw data.
"""

from __future__ import annotations

import warnings
from dataclasses import dataclass, field

import numpy as np

from ..config import CastParams
from ..models import RawADCP
from .ingest import DualHead

_ERR_COMP = 3      # earth coord component 4 = error velocity
_PG_FIELD = 3      # percent-good field 4 = % good 4-beam


@dataclass
class ScreenResult:
    counts: dict[str, int] = field(default_factory=dict)
    tilt_down: np.ndarray | None = None        # [nens] deg
    tiltd_down: np.ndarray | None = None        # [nens] deg
    good_down: np.ndarray | None = None         # [ncell, nens] bool (cell passes edits)
    good_up: np.ndarray | None = None
    warnings: list[str] = field(default_factory=list)


def tilt_series(head: RawADCP) -> tuple[np.ndarray, np.ndarray]:
    """Return (tilt, |d tilt|) per ensemble [deg], per ``loadrdi.m``."""
    pit = np.radians(head.pitch)
    rol = np.radians(head.roll)
    tilt = np.degrees(np.arcsin(np.sqrt(np.sin(pit) ** 2 + np.sin(rol) ** 2)))
    tiltd = np.sqrt(_centred_absdiff(head.roll) ** 2 + _centred_absdiff(head.pitch) ** 2)
    return tilt, tiltd


def _centred_absdiff(x: np.ndarray) -> np.ndarray:
    """Centred absolute difference with zero-padded ends (matches MATLAB)."""
    fwd = np.abs(np.diff(np.concatenate(([0.0], x))))    # back-difference into k
    bwd = np.abs(np.diff(np.concatenate((x, [0.0]))))    # forward-difference out of k
    return (fwd + bwd) / 2.0


def _cell_good(head: RawADCP, pglim: float, elim: float) -> tuple[np.ndarray, int, int]:
    """Good-cell mask + (pg_removed, errvel_removed) for one head."""
    pg4 = head.pct_good[_PG_FIELD]
    ev = head.vel[_ERR_COMP]
    pg_bad = pg4 < pglim
    ev_bad = (~pg_bad) & (np.abs(ev) > elim)           # errvel counted only on pg-survivors
    good = ~(pg_bad | ev_bad)
    return good, int(pg_bad.sum()), int(ev_bad.sum())


def screen(dh: DualHead, params: CastParams | None = None) -> ScreenResult:
    """Apply the loadrdi threshold edits and return masks + golden-comparable counts."""
    p = params or dh.params or CastParams(station=dh.station)
    res = ScreenResult()

    good_d, pg_d, ev_d = _cell_good(dh.down, p.pglim, p.elim)
    res.good_down = good_d
    res.counts["pg_removed_down"] = pg_d

    pg_u = ev_u = 0
    if dh.has_up:
        good_u, pg_u, ev_u = _cell_good(dh.up, p.pglim, p.elim)
        res.good_up = good_u
        res.counts["pg_removed_up"] = pg_u

    res.counts["errvel_removed"] = ev_d + ev_u          # golden reports combined

    # tilt (down-looker) -> ensemble rejection
    tilt, tiltd = tilt_series(dh.down)
    res.tilt_down, res.tiltd_down = tilt, tiltd
    n_t1 = int(np.sum(tilt > p.tiltmax[0]))
    n_t2 = int(np.sum(tiltd > p.tiltmax[1]))
    res.counts["tilt_removed_gt22"] = n_t1
    res.counts["tilt_removed_deriv4"] = n_t2
    bad_ens = (tilt > p.tiltmax[0]) | (tiltd > p.tiltmax[1])
    if res.good_down is not None:
        res.good_down[:, bad_ens] = False
    if n_t1 > len(tilt) * 0.1:
        res.warnings.append(f" {int(n_t1*100/len(tilt))}%  tilt > {int(p.tiltmax[0])} ")

    # horizontal speed (indicative; see module note) + middle-hour warning
    res.counts["hspeed_removed_approx"] = _hspeed_count(dh, p.vlim)
    n_mid = sum(_middle_hour_fast(h, p.vlim) for h in (dh.down, dh.up) if h is not None)
    res.counts["hspeed_mid_hour_approx"] = n_mid
    if n_mid > 10:
        res.warnings.append(
            f"**  found  {n_mid}  horizontal velocities > {int(p.vlim)}m/s in middle hour of cast")
    return res


# near-field / far-field error-velocity ratio: a rigid target hung below the package
# (e.g. a corer on a short cable) reads the package's motion, not the water, doubling
# |errvel| at its fixed range. MORIA scan: contaminated casts 1.69-2.35, clean <= 1.41.
NEARFIELD_RANGE = (18.0, 42.0)      # m, down-looker ranges a short-cable device occupies
NEARFIELD_FAR_MIN = 50.0            # m, far-field baseline starts here
NEARFIELD_WARN_RATIO = 1.7


def nearfield_errvel_ratio(dh: DualHead) -> float:
    """Hottest near-field down-looker bin's median|errvel| over the far field.

    A non-destructive hung-device detector: > ~1.7 means something rigid is reflecting
    at short range below the package (see the MORIA monocorer block 03-28). The device
    occupies only one or two bins, so the near-field statistic is the MAX of the
    per-bin medians (a pooled median would dilute it with clean neighbours). It removes
    no data -- pair it with ``edit_nearfield_dn_bins`` to actually mask the cells
    (always the user's explicit call; :func:`nearfield_hot_bin` names the bin to mask).
    """
    return _nearfield_stats(dh)[0]


# per-bin "elevated" cut for the WARN's mask suggestion. Measured on MORIA: the clean
# deep cast 80 tops out at 1.50 (bin 4), the contaminated band's affected bins span
# 1.79-2.52 (BOTH the 26 m device cell and the 34 m echo-tail cell) -- 1.6 separates.
NEARFIELD_BIN_RATIO = 1.6


def nearfield_elevated_bins(dh: DualHead,
                            ratio_min: float = NEARFIELD_BIN_RATIO,
                            ) -> list[tuple[int, float]]:
    """Near-field down-looker bins with elevated |errvel|: ``[(1-based bin, range m)]``.

    Feeds the QA WARN's actionable note: these are exactly the bins to pass to
    ``--nearfield-dn-bins``. Data-driven, so it lists the device cell AND its echo
    tail without guessing geometry (on MORIA both 26 m + 34 m cells are elevated;
    the hottest one is the tail). Empty when nothing is elevated or no statistics.
    """
    return [(b, rng) for b, rng, r in _nearfield_stats(dh)[1] if r > ratio_min]


def _nearfield_stats(dh: DualHead) -> tuple[float, list[tuple[int, float, float]]]:
    """``(max near/far ratio, [(1-based bin, range m, per-bin ratio), ...])``."""
    d = dh.down
    ev = np.abs(d.vel[_ERR_COMP].astype(float))
    rng = dh.bin_depth(d)
    near = (rng >= NEARFIELD_RANGE[0]) & (rng < NEARFIELD_RANGE[1])
    far = rng >= NEARFIELD_FAR_MIN
    if near.sum() < 1 or far.sum() < 2:
        return float("nan"), []
    with warnings.catch_warnings():
        warnings.simplefilter("ignore", RuntimeWarning)    # all-NaN bins on sparse casts
        per_bin = np.nanmedian(ev[near], axis=1)
        m_far = np.nanmedian(ev[far])
    if not np.isfinite(per_bin).any() or not np.isfinite(m_far) or m_far <= 0:
        return float("nan"), []
    detail = [(int(b) + 1, float(r), float(v / m_far))
              for b, r, v in zip(np.flatnonzero(near), rng[near], per_bin, strict=True)
              if np.isfinite(v)]
    return float(np.nanmax(per_bin) / m_far), detail


def _hspeed_count(dh: DualHead, vlim: float) -> int:
    n = 0
    for head in (dh.down, dh.up):
        if head is None:
            continue
        pg4 = head.pct_good[_PG_FIELD]
        ev = head.vel[_ERR_COMP]
        bad = (pg4 < 50) | (np.abs(ev) > 0.2)        # pg + errvel edits precede hspeed
        u = np.where(bad, np.nan, head.vel[0])
        v = np.where(bad, np.nan, head.vel[1])
        n += int(np.sum(np.sqrt(u ** 2 + v ** 2) > vlim))
    return n


def _middle_hour_fast(head: RawADCP, vlim: float) -> int:
    """Count fast horizontal velocities in the middle hour (skip +-1200 s of ends)."""
    nens = head.n_ens
    t = (head.time - head.time[0]) / np.timedelta64(1, "s")
    enstime = (np.nanmax(t) - np.nanmin(t)) / max(nens, 1)
    if not np.isfinite(enstime) or enstime <= 0:
        return 0
    skip = int(round(1200 / enstime))
    spd = np.sqrt(head.vel[0] ** 2 + head.vel[1] ** 2)   # [ncell, nens]
    cols = np.arange(nens)
    sel = (cols > skip) & (cols < nens - skip)
    return int(np.sum(spd[:, sel] > vlim))
