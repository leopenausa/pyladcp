"""Phase 3 — attitude / engineering health (tilt, heading, transmit voltage).

All quantities here derive from raw per-ping sensors, independent of the inverse:

* tilt statistics (max, mean, fraction over limit) — exact, from the down-looker.
* heading rotation span over the cast.
* transmit voltage -> battery estimate (legacy ``battery.m``: ``0.33 * xmv`` for an
  unknown CPU board). The golden ``p.xmv`` is the *in-cast* mean (deck spikes excluded);
  until surface/bottom trim lands we use the median, which is robust to those spikes.

Dual-head mounting offset: the *precise* offsets the golden records
(``up_dn_comp_off``, ``up_dn_pit_rol_comp_off``, pitch/roll offsets) are computed in
``prepinv.m`` on **super-ensembles** (a velocity-stage construct). Here we report a
raw-ping tilt-based *estimate* as an acquisition consistency check; it agrees with the
golden to ~1-2 deg but is explicitly not the bit-exact velocity-stage value.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
from scipy.optimize import minimize

from ..models import Metric, Status
from .ingest import DualHead
from .screen import tilt_series

_BATTERY_FACTOR = 0.33          # legacy default guess (unknown CPU board)
_BATTERY_LOW = 40.0


@dataclass
class AttitudeSummary:
    tilt_max: float
    tilt_mean: float
    tilt_pct_over: float            # % ensembles over tiltmax[0]
    heading_span: float             # deg of rotation coverage
    xmv_median: float
    battery_v: float
    dual_head_offset_est: float | None   # raw-ping tilt-based estimate [deg]
    pitch_offset_est: float | None
    roll_offset_est: float | None


def _uvrot(u, v, rot):
    r = -np.radians(rot)
    cr, sr = np.cos(r), np.sin(r)
    return u * cr - v * sr, u * sr + v * cr


def dual_head_offset(dh: DualHead) -> tuple[float, float, float]:
    """Raw-ping tilt-based heading offset + pitch/roll offsets [deg] (estimate).

    Mirrors ``checktilt``/``prepinv`` on raw, start-aligned pings instead of
    super-ensembles: minimise the variance of (rotated down-tilt - up-tilt), then take
    the residual mean pitch/roll differences after rotation.
    """
    n = min(dh.down.n_ens, dh.up.n_ens)
    rol_d, pit_d = dh.down.roll[:n], dh.down.pitch[:n]
    rol_u, pit_u = dh.up.roll[:n], dh.up.pitch[:n]

    def cost(rot):
        r21, p21 = _uvrot(rol_d, pit_d, rot[0])
        dr, dp = r21 - rol_u, p21 - pit_u
        return np.sum((dr - dr.mean()) ** 2 + (dp - dp.mean()) ** 2)

    hoff2 = float(minimize(cost, x0=[0.0], method="Nelder-Mead").x[0])
    rol_u_rot, pit_u_rot = _uvrot(rol_u, pit_u, -hoff2)
    roll_off = float(np.mean(rol_d - rol_u_rot))
    pitch_off = float(np.mean(pit_d - pit_u_rot))
    return hoff2, pitch_off, roll_off


def attitude_summary(dh: DualHead) -> AttitudeSummary:
    d = dh.down
    tilt, _ = tilt_series(d)
    tiltmax = (dh.params.tiltmax[0] if dh.params else 22.0)
    hdg = d.heading[np.isfinite(d.heading)]
    xmv_med = float(np.nanmedian(d.xmit_voltage)) if d.xmit_voltage is not None else np.nan

    off = pit = rol = None
    if dh.has_up:
        off, pit, rol = dual_head_offset(dh)

    return AttitudeSummary(
        tilt_max=float(np.nanmax(tilt)),
        tilt_mean=float(np.nanmean(tilt)),
        tilt_pct_over=float(100.0 * np.mean(tilt > tiltmax)),
        heading_span=float(np.nanmax(hdg) - np.nanmin(hdg)) if hdg.size else np.nan,
        xmv_median=xmv_med,
        battery_v=_BATTERY_FACTOR * xmv_med,
        dual_head_offset_est=off,
        pitch_offset_est=pit,
        roll_offset_est=rol,
    )


def attitude_metrics(a: AttitudeSummary) -> list[Metric]:
    # WARN at most: the V estimate is 0.33*xmv with a board-dependent factor legacy
    # battery.m only guesses at, and pack nominal voltage varies by cruise -- a low
    # *estimate* is an engineering heads-up, not grounds to FAIL an otherwise good cast
    # (it failed 17/40 MORIA stations whose data were fine).
    bat_status = Status.OK if a.battery_v > _BATTERY_LOW else Status.WARN
    tilt_status = Status.OK if a.tilt_pct_over < 1 else Status.WARN
    out = [
        Metric("tilt_max", round(a.tilt_max, 2), "deg", tilt_status,
               source_stage="qa.attitude",
               note=f"mean {a.tilt_mean:.1f} deg, {a.tilt_pct_over:.1f}% over limit"),
        Metric("battery", round(a.battery_v, 1), "V", bat_status,
               threshold={"low": _BATTERY_LOW},
               source_stage="qa.attitude",
               note="0.33*median(xmv) (matches golden p.xmv to 0.02 V); conversion "
                    "factor is board-dependent, so this can WARN but never FAIL"),
        Metric("heading_rotation", round(a.heading_span, 0), "deg", Status.OK,
               source_stage="qa.attitude"),
    ]
    if a.dual_head_offset_est is not None:
        out.append(Metric(
            "dual_head_offset_est", round(a.dual_head_offset_est, 2), "deg", Status.OK,
            source_stage="qa.attitude",
            note=(f"tilt-based estimate (raw pings); pitch {a.pitch_offset_est:.2f}, "
                  f"roll {a.roll_offset_est:.2f}; exact value is velocity-stage (prepinv)")))
    return out
