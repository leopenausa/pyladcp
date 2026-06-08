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
from .inverse import VelocityResult


def consistency_checks(r: VelocityResult) -> list[Metric]:
    """Return the ``checkinv`` consistency metrics derivable from ``r``.

    * **velocity_error_vs_noise** — median profile uncertainty vs the per-cell fit-residual
      noise floor; the formal error should be of the same order (not far below it).
    * **bottom_track_consistency** — bias between our own and the RDI firmware bottom track;
      legacy flags |bias| > 0.1 m/s.
    * **sadcp_consistency** — RMS of (LADCP − ship-ADCP) over their shared depth range.
    """
    out: list[Metric] = []

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

    return out
