"""Phase 3 — profiling range from correlation ("range of good data").

Ports the per-beam range calculation in ``plotraw.m``: reduce correlation magnitude over
ensembles to a per-bin profile, then for each beam find the bin whose correlation is
closest to 30% of the strongest bin-1 correlation; that bin's distance is the usable
range. Validated exactly against the golden MORIA-80 ``p`` struct:

    dn_range = [170.11, 170.11, 170.11, 170.11]
    up_range = [170.11, 154.11, 162.11, 170.11]   <- up beams 2 & 3 sub-nominal

A beam whose range falls below ``warn_frac`` of the best beam's range is flagged
sub-nominal (WARN) even though it is not broken — exactly the kind of mild acquisition
issue this stage exists to surface.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np

from ..models import Metric, RawADCP, Status

_CORR_FRAC = 0.30           # legacy threshold: 30% of peak bin-1 correlation
_WARN_FRAC = 0.96           # flag beams shorter than 96% of the best beam's range


@dataclass
class RangeResult:
    head: str
    beam_range: np.ndarray      # [4] m
    flags: list[str]            # per-beam "ok"|"sub-nominal"
    status: Status


def profiling_range(head: RawADCP, warn_frac: float = _WARN_FRAC) -> RangeResult:
    cm = np.nanmedian(head.corr, axis=2)        # [4, nbin]
    dist0 = head.meta.get("dist_first_m", head.blank_m + head.cell_m / 2.0)
    z = dist0 + np.arange(head.n_cells) * head.cell_m
    thr = _CORR_FRAC * np.nanmax(cm[:, 0])      # 30% of strongest bin-1 correlation
    rng = np.array([z[np.argmin(np.abs(cm[i] - thr))] for i in range(4)])

    rmax = rng.max()
    flags, status = [], Status.OK
    for r in rng:
        if r < warn_frac * rmax:
            flags.append("sub-nominal")
            status = status if status == Status.FAIL else Status.WARN
        else:
            flags.append("ok")
    return RangeResult(head=head.facing, beam_range=rng, flags=flags, status=status)


def range_metric(rr: RangeResult) -> Metric:
    sub = [f"beam {i+1} short ({rr.beam_range[i]:.0f} m)"
           for i in range(4) if rr.flags[i] != "ok"]
    return Metric(
        name=f"profiling_range_{rr.head}",
        value=[round(float(x), 2) for x in rr.beam_range],
        unit="m",
        status=rr.status,
        threshold={"corr_frac": _CORR_FRAC, "warn_frac": _WARN_FRAC},
        source_stage="qa.range",
        note=("; ".join(sub) if sub else "all beams nominal"),
    )
