"""Phase 3 — assemble the acquisition-QA report.

Runs the core-health diagnostics over a :class:`DualHead` and collects them into a
:class:`QCMetrics` object (machine-readable) plus a short human-readable summary. This is
the QA-first deliverable: a single call that says whether a cast's *raw data* is sound,
independent of any velocity solution.
"""

from __future__ import annotations

from ..config import CastParams
from ..models import CTDTimeSeries, Metric, QCMetrics, Status
from .attitude import attitude_metrics, attitude_summary
from .beams import beam_health, beam_metric
from .bottom import bottom_metric, detect_bottom
from .depth import synchronize
from .ingest import DualHead
from .range import profiling_range, range_metric
from .screen import screen


def assess(dh: DualHead, params: CastParams | None = None,
           ctd: CTDTimeSeries | None = None) -> QCMetrics:
    """Compute the acquisition-QA metric set for one cast.

    When a CTD time-series is supplied, also performs CTD<->LADCP synchronization,
    package-depth and seabed detection (depth/bottom metrics).
    """
    p = params or dh.params
    qc = QCMetrics(station=dh.station or "")

    # --- editing / screening counts ---
    sr = screen(dh, p)
    qc.warnings.extend(sr.warnings)
    for k, v in sr.counts.items():
        qc.add(Metric(f"edit_{k}", v, "cells/ensembles", Status.OK, source_stage="qa.screen"))

    # --- per-head beam + range health ---
    for head in (dh.down, dh.up):
        if head is None:
            continue
        qc.add(beam_metric(beam_health(head)))
        qc.add(range_metric(profiling_range(head)))

    # --- attitude / engineering ---
    for m in attitude_metrics(attitude_summary(dh)):
        qc.add(m)

    # --- depth / bottom (requires CTD) ---
    if ctd is not None:
        sync = synchronize(dh, ctd)
        sync_status = Status.OK if sync.corr > 0.9 else Status.WARN
        qc.add(Metric("ctd_sync_corr", round(sync.corr, 3), "", sync_status,
                      source_stage="qa.depth",
                      note=f"lag {sync.lag} s; max package depth {sync.maxdepth:.0f} m"))
        qc.add(bottom_metric(detect_bottom(dh, sync, ctd=ctd)))

    return qc


def text_report(qc: QCMetrics) -> str:
    """Render a compact, operator-readable QA summary."""
    lines = [
        f"LADCP acquisition QA — {qc.station}",
        f"overall: {qc.overall_status.value.upper()}",
        "-" * 56,
    ]
    icon = {Status.OK: "ok  ", Status.WARN: "WARN", Status.FAIL: "FAIL"}
    for name, m in qc.metrics.items():
        val = m.value
        line = f"[{icon[m.status]}] {name:28s} {val} {m.unit}".rstrip()
        if m.note:
            line += f"\n           {m.note}"
        lines.append(line)
    if qc.warnings:
        lines.append("-" * 56)
        lines.append("warnings:")
        lines.extend(f"  ! {w.strip()}" for w in qc.warnings)
    return "\n".join(lines)
