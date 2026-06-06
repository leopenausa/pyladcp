"""Phase 3 — beam performance (echo-amplitude signal-to-noise).

Ports the ``checkbeam`` logic embedded in ``plotraw.m``: for each of the 4 beams,
remove the source level (mean over bins), estimate the noise floor from the far half of
the profile, and form a signal-to-noise as ``mean(first 2 bins) - noise``. Beams are
flagged relative to the strongest beam:

    broken  : s2n < 0.50 * max(s2n)     (FAIL)
    bad     : s2n < 0.65 * max(s2n)     (FAIL)
    weak    : s2n < 0.80 * max(s2n)     (WARN)

Validated: the integer percentages ``round(s2n / max(s2n) * 100)`` reproduce the golden
MORIA-80 Figure 2 values exactly (down [98,95,100,96], up [100,98,100,98]).
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np

from ..models import Metric, RawADCP, Status
from .stats import medianan

_BROKEN, _BAD, _WEAK = 0.50, 0.65, 0.80


@dataclass
class BeamHealth:
    head: str                       # "down" | "up"
    s2n: np.ndarray                 # [4] signal-to-noise (echo-amplitude units)
    pct: np.ndarray                 # [4] round(s2n/max*100)
    flags: list[str]                # per-beam: "ok"|"weak"|"bad"|"broken"
    status: Status


def beam_echo_median(head: RawADCP) -> np.ndarray:
    """Per-beam, per-bin echo amplitude reduced over ensembles via ``medianan`` (na=0)."""
    nb, nc, _ = head.echo.shape
    out = np.empty((nb, nc))
    for b in range(nb):
        for c in range(nc):
            out[b, c] = medianan(head.echo[b, c, :], 0)
    return out


def beam_health(head: RawADCP) -> BeamHealth:
    ea = beam_echo_median(head)                 # [4, nbin]
    nbin = ea.shape[1]
    iend = np.arange(nbin // 2 - 1, nbin)       # legacy fix(bl/2):bl, 1-based -> 0-based
    s2n = np.empty(4)
    for i in range(4):
        t = ea[i] - np.nanmean(ea[i])           # remove source level
        s2n[i] = np.nanmean(t[:2]) - medianan(t[iend], 2)
    smax = np.nanmax(s2n)
    pct = np.round(s2n / smax * 100).astype(int)

    flags, status = [], Status.OK
    for v in s2n:
        r = v / smax
        if r < _BROKEN:
            flags.append("broken"); status = Status.FAIL
        elif r < _BAD:
            flags.append("bad"); status = Status.FAIL
        elif r < _WEAK:
            flags.append("weak")
            status = status if status == Status.FAIL else Status.WARN
        else:
            flags.append("ok")
    return BeamHealth(head=head.facing, s2n=s2n, pct=pct, flags=flags, status=status)


def beam_metric(bh: BeamHealth) -> Metric:
    note = ", ".join(f"b{i+1}:{bh.pct[i]}%({bh.flags[i]})" for i in range(4))
    bad = [f"beam {i+1} {bh.flags[i]}" for i in range(4) if bh.flags[i] != "ok"]
    return Metric(
        name=f"beam_performance_{bh.head}",
        value=bh.pct.tolist(),
        unit="% of best beam S/N",
        status=bh.status,
        threshold={"weak": _WEAK, "bad": _BAD, "broken": _BROKEN},
        source_stage="qa.beams",
        note=note + (" | " + "; ".join(bad) if bad else ""),
    )
