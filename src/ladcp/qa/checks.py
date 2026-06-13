"""Post-inversion consistency checks (port of LDEO_IX ``checkinv``).

``checkinv`` cross-checks the finished solution against its own error estimates and
constraints: the profile error should sit near the data noise floor, the bottom-track
reference should agree between independent estimates, and an independent ship-ADCP profile
should match the LADCP velocity. Here each check becomes a traffic-light :class:`Metric`
on the QA scorecard rather than a console print.
"""

from __future__ import annotations

import numpy as np

from ..models import Metric, Status
from ..plots.sadcp_figure import sadcp_rms_discrepancy
from .inverse import _SURFACE_FILL_MAX, VelocityResult

_SHIP_DRIFT_WARN = 300.0          # net in-cast GPS displacement [m] above which the ship was
#                                   not on station (barotropic reference corrected but layback
#                                   uncertainty remains)


def consistency_checks(r: VelocityResult) -> list[Metric]:
    """Return the ``checkinv`` consistency metrics derivable from ``r``.

    * **velocity_error_vs_noise** — median profile uncertainty vs the per-cell fit-residual
      noise floor; the formal error should be of the same order (not far below it).
    * **bottom_track_consistency** — bias between our own and the RDI firmware bottom track;
      legacy flags |bias| > 0.1 m/s.
    * **sadcp_consistency** — RMS of (LADCP − ship-ADCP) over their shared depth range.
    * **profile_surface_coverage** — flags an unsampled upper water column (the ADCP began
      mid-cast: a recording gap or a fragmented file), so the profile starts far below the surface.
    """
    out: list[Metric] = []

    z, nvel = r.vp.z, r.vp.nvel
    sampled = np.where(nvel > 0)[0]
    if sampled.size:
        z_top = float(z[sampled[0]])
        if z_top > _SURFACE_FILL_MAX:
            out.append(Metric(
                "profile_surface_coverage", round(z_top, 0), "m", Status.WARN,
                source_stage="qa.checks",
                note=(f"no LADCP velocity above {z_top:.0f} m -- the ADCP began mid-cast "
                      f"(recording gap or fragmented file); the upper {z_top:.0f} m is "
                      f"unsampled, not zero velocity")))

    if r.drift is not None and r.drift.ship_e.size > 1:
        net = float(np.hypot(r.drift.ship_e[-1] - r.drift.ship_e[0],
                             r.drift.ship_n[-1] - r.drift.ship_n[0]))
        if net > _SHIP_DRIFT_WARN:
            out.append(Metric(
                "ship_drift", round(net, 0), "m", Status.WARN,
                source_stage="qa.checks",
                note=(f"ship moved {net:.0f} m during the cast (not on station); the barotropic "
                      f"reference uses this GPS drift, but a moving ship on a long wire leaves "
                      f"residual layback uncertainty in the absolute velocity")))

    noise = r.resid_rms
    uerr = float(np.nanmedian(r.vp.uerr)) if r.vp.uerr.size else np.nan
    if np.isfinite(uerr) and np.isfinite(noise) and noise > 0:
        ratio = uerr / noise
        status = Status.OK if ratio <= 2.5 else Status.WARN
        out.append(Metric(
            "velocity_error_vs_noise", round(ratio, 2), "x", status,
            source_stage="qa.checks",
            note=f"profile err {uerr:.3f} vs noise floor {noise:.3f} m/s"))

    b = r.btrk
    if b is not None and b.n_rdi > 0 and b.n_own > 0:
        bias = float(np.nanmax(np.abs([b.u_bias, b.v_bias])))
        status = Status.OK if bias <= 0.1 else Status.WARN
        out.append(Metric(
            "bottom_track_consistency", round(bias, 3), "m/s", status,
            source_stage="qa.checks",
            note=(f"own−RDI bias u {b.u_bias:+.3f} v {b.v_bias:+.3f}; "
                  f"own n {b.n_own}, RDI n {b.n_rdi}")))

    if r.sadcp is not None and len(r.sadcp):
        rms = sadcp_rms_discrepancy(r)
        if np.isfinite(rms):
            status = Status.OK if rms <= 0.1 else Status.WARN
            out.append(Metric(
                "sadcp_consistency", round(rms, 3), "m/s", status,
                source_stage="qa.checks",
                note=f"RMS(LADCP − ship-ADCP) over shared depths, {len(r.sadcp)} bins"))

    # shear-vs-inverse disagreement (legacy getshear2.m:143-158): the inverse and shear
    # method are independent estimators of the baroclinic shear; legacy WARNs (and has
    # already inflated the delivered uerr) when they disagree by more than 1.5x the formal
    # error AND by more than 0.1 m/s in absolute terms.
    si = r.shear_inverse
    if si is not None:
        uvds, mean_uerr = si
        if np.isfinite(uvds) and np.isfinite(mean_uerr) and mean_uerr > 0:
            status = Status.WARN if (uvds > 1.5 * mean_uerr and uvds > 0.1) else Status.OK
            inflated = uvds > mean_uerr
            out.append(Metric(
                "shear_vs_inverse_consistency", round(uvds, 3), "m/s", status,
                source_stage="qa.checks",
                note=(f"baroclinic shear vs inverse disagree by {uvds:.3f} m/s "
                      f"(mean formal err {mean_uerr:.3f} m/s"
                      + ("; delivered uerr inflated to uvds/1.5 per legacy getshear2)"
                         if inflated else "; within formal error)"))))

    return out
