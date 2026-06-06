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
