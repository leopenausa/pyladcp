"""Phase 3 — assemble the acquisition-QA report.

Runs the core-health diagnostics over a :class:`DualHead` and collects them into a
:class:`QCMetrics` object (machine-readable) plus a short human-readable summary. This is
the QA-first deliverable: a single call that says whether a cast's *raw data* is sound,
independent of any velocity solution.
"""

from __future__ import annotations

import numpy as np

from ..config import CastParams
from ..models import CoordFrame, CTDTimeSeries, Metric, QCMetrics, Status
from .attitude import attitude_metrics, attitude_summary
from .beams import beam_health, beam_metric
from .bottom import bottom_metric, detect_bottom
from .depth import synchronize, water_window
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

    # --- coordinate-frame guard (velocity/sync/bottom require an earth frame) ---
    coord_metrics, earth = _coord_frame_metrics(dh)
    for m in coord_metrics:
        qc.add(m)

    # --- depth / bottom (requires CTD + earth-frame velocities) ---
    if ctd is not None and earth:
        sync = synchronize(dh, ctd)
        sync_status = Status.OK if sync.corr > 0.9 else Status.WARN
        qc.add(Metric("ctd_sync_corr", round(sync.corr, 3), "", sync_status,
                      source_stage="qa.depth",
                      note=f"offset {sync.lag} s; max package depth {sync.maxdepth:.0f} m"))
        for m in _sync_quality_metrics(dh, sync):
            qc.add(m)
        for m in _depth_sanity_metrics(sync):
            qc.add(m)
        qc.add(bottom_metric(detect_bottom(dh, sync, ctd=ctd)))

    return qc


def _coord_frame_metrics(dh: DualHead) -> tuple[list[Metric], bool]:
    """Per-head ``coord_frame`` metrics + whether every head is earth-frame.

    Velocity/sync/bottom read ``vel[0..3]`` as ``(u, v, w, e)``, valid only in the earth frame;
    a beam-coordinate head (no beam->earth transform yet) would yield silent garbage, so it is
    flagged WARN and ``earth`` returns False to gate the velocity-dependent QA.
    """
    metrics: list[Metric] = []
    earth = True
    for label, head in (("down", dh.down), ("up", dh.up)):
        if head is None:
            continue
        is_earth = head.coord_frame == CoordFrame.EARTH
        earth = earth and is_earth
        metrics.append(Metric(
            f"coord_frame_{label}", head.coord_frame.value, "",
            Status.OK if is_earth else Status.WARN, source_stage="qa.ingest",
            note=None if is_earth else
            "velocities require a beam->earth transform (not yet implemented); "
            "acquisition metrics valid, velocity/sync/bottom NOT computed"))
    return metrics, earth


def _sync_quality_metrics(dh: DualHead, sync) -> list[Metric]:
    """Fail-loud guards that the CTD<->LADCP synchronization actually landed on the cast.

    The single most dangerous silent failure (FDCCC1_001) was a mis-sync that mapped the CTD
    depth onto the ADCP's on-deck pings: every metric still computed, but the in-water depth
    window contained no valid velocity and the profile collapsed to empty. These two flags make
    that loud. Pure flags -- they change no result.

    * ``ctd_sync_locate`` -- the coarse cross-correlation peak (:func:`depth._coarse_offset`).
      A low value means the cast could not be confidently located in the recording.
    * ``ctd_sync_coverage`` -- the fraction of in-water pings that carry a valid velocity. Near
      zero means the depth window and the data disagree (a mis-sync), so the solve would be
      empty/garbage.
    """
    metrics: list[Metric] = []
    cs = sync.coarse_score
    if np.isfinite(cs):
        metrics.append(Metric(
            "ctd_sync_locate", round(float(cs), 2), "",
            Status.WARN if cs < 0.5 else Status.OK, source_stage="qa.depth",
            note=("CTD cast could not be confidently located in the ADCP record (weak vertical-"
                  "velocity cross-correlation); the depth/velocity mapping may be wrong"
                  if cs < 0.5 else None)))
    z = np.asarray(sync.z_on_ping, float)
    n = dh.down.n_ens
    i0, i1 = water_window(z[:n])
    inwin = np.zeros(n, dtype=bool)
    inwin[i0:i1 + 1] = True
    good = (np.isfinite(dh.down.vel[0]).sum(axis=0) >= 3)[:n]
    denom = int(inwin.sum())
    frac = float((good & inwin).sum() / denom) if denom else 0.0
    metrics.append(Metric(
        "ctd_sync_coverage", round(frac, 2), "fraction",
        Status.WARN if frac < 0.2 else Status.OK, source_stage="qa.depth",
        note=("in-water depth window carries almost no valid velocity: CTD/LADCP likely "
              "mis-synced -- the profile would be empty or garbage" if frac < 0.2 else None)))
    return metrics


def _depth_sanity_metrics(sync) -> list[Metric]:
    """Start-depth and shallow-cast sanity flags (legacy getdpthi shallow/surface guards).

    Pure flags -- they change no result. ``start_depth`` WARNs when the first in-water package
    depth exceeds 50 m (legacy ``getdpthi.m`` warns on ``d.z(1)<-50``): a clipped cast or a
    CTD/LADCP mis-sync. ``shallow_cast`` WARNs below 100 m: legacy disables surface detection
    there and the thin water column yields few super-ensembles (weakly constrained solve).
    """
    metrics: list[Metric] = []
    i0, _ = water_window(sync.z_on_ping)
    z_start = float(sync.z_on_ping[i0])
    if np.isfinite(z_start):
        metrics.append(Metric(
            "start_depth", round(z_start, 0), "m",
            Status.WARN if z_start > 50.0 else Status.OK, source_stage="qa.depth",
            note=("first in-water package depth > 50 m: cast may be clipped or CTD/LADCP "
                  "mis-synced" if z_start > 50.0 else None)))
    if sync.maxdepth < 100.0:
        metrics.append(Metric(
            "shallow_cast", round(sync.maxdepth, 0), "m", Status.WARN, source_stage="qa.depth",
            note="shallow cast (<100 m): surface detection not performed; thin water column "
                 "-> few super-ensembles, weakly constrained solve"))
    return metrics


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
