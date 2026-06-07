"""Per-station QA PDF report.

Page 1 is a traffic-light scorecard (green / amber / red per diagnostic). Each following
page renders a QA figure **as vector** (crisp, not a rasterised screenshot) into the top
sub-figure of a clean A4 page, with a "what to expect" interpretation in a reserved
caption sub-figure below — so nothing clips. The individual PNG figures are also written
out independently at high resolution.
"""

from __future__ import annotations

import textwrap
from pathlib import Path

from ..models import CTDTimeSeries, QCMetrics, Status
from ..qa.ingest import DualHead
from ..qa.inverse import VelocityResult
from .alignment import alignment_figure
from .btrack_figure import btrack_figure
from .depth_figure import depth_figure
from .drift_figure import drift_figure
from .edit_figure import edit_figure
from .error_figure import error_figure
from .inverse_figure import inverse_diagnostics_figure
from .raw_dashboard import raw_dashboard
from .sadcp_figure import sadcp_figure, sadcp_rms_discrepancy
from .shear_figure import shear_figure
from .velocity_figure import velocity_figure

_COLOR = {Status.OK: "#27ae60", Status.WARN: "#f1c40f", Status.FAIL: "#e74c3c"}
_LABEL = {Status.OK: "OK", Status.WARN: "CHECK", Status.FAIL: "FAIL"}

_HEADLINE = [
    "beam_performance_down", "beam_performance_up",
    "profiling_range_down", "profiling_range_up",
    "tilt_max", "battery", "heading_rotation",
    "dual_head_offset_est", "ctd_sync_corr", "bottom_depth",
]
_A4 = (8.27, 11.69)
_CONSISTENCY = ["velocity_error_vs_noise", "bottom_track_consistency", "sadcp_consistency"]


def build_report(dh: DualHead, qc: QCMetrics, outdir: str, station: str,
                 ctd: CTDTimeSeries | None = None,
                 velocity: VelocityResult | None = None,
                 figdir: str | None = None) -> dict[str, str]:
    """Write high-res PNGs + a vector `<station>_report.pdf`. Returns paths written.

    When ``velocity`` (a :class:`~ladcp.qa.inverse.VelocityResult`) is supplied, the velocity
    profile / shear / inversion-diagnostics pages are appended after the acquisition figures.

    ``figdir`` redirects the standalone PNGs into a separate directory (e.g. a ``figures/``
    subdir) while ``<station>_report.pdf`` stays in ``outdir``; default keeps both together.
    """
    import matplotlib.pyplot as plt
    from matplotlib.backends.backend_pdf import PdfPages

    out = Path(outdir)
    out.mkdir(parents=True, exist_ok=True)
    figout = Path(figdir) if figdir else out
    figout.mkdir(parents=True, exist_ok=True)

    # (draw fn into a given figure, png name, page title, caption)
    pages = [(lambda f: raw_dashboard(dh, fig=f), f"{station}_raw.png",
              "Figure 2 — Raw-data overview", _caption_raw(qc))]
    if dh.has_up:
        pages.append((lambda f: alignment_figure(dh, fig=f), f"{station}_alignment.png",
                      "Figure 6 — Dual-head alignment", _caption_align(qc)))
    if ctd is not None:
        pages.append((lambda f: depth_figure(dh, ctd, fig=f), f"{station}_depth.png",
                      "Figure 4 — Surface & seabed detection", _caption_depth(qc)))
        pages.append((lambda f: edit_figure(dh, ctd, fig=f), f"{station}_edit.png",
                      "Figure 14 — Data editing", _caption_edit(qc)))
    if velocity is not None:
        v = velocity
        pages.append((lambda f: velocity_figure(v.vp, bottom=v.bp, fig=f),
                      f"{station}_velocity.png",
                      "Figure 1 — Ocean velocity profile", _caption_velocity(v)))
        pages.append((lambda f: shear_figure(v.shear, fig=f), f"{station}_shear.png",
                      "Figure 3 — Vertical shear", _caption_shear(v)))
        pages.append((lambda f: inverse_diagnostics_figure(v, fig=f),
                      f"{station}_inverse.png",
                      "Figure 12 — Inversion diagnostics", _caption_inverse(v)))
        if v.err is not None:
            pages.append((lambda f: error_figure(v, fig=f), f"{station}_error.png",
                          "Figure 3 — Super-ensemble error", _caption_error(v)))
        if v.drift is not None:
            pages.append((lambda f: drift_figure(v, fig=f), f"{station}_drift.png",
                          "Figure 8 — Ship & CTD drift", _caption_drift(v)))
        if v.btrk is not None and v.btrk.n_own:
            pages.append((lambda f: btrack_figure(v, fig=f), f"{station}_btrack.png",
                          "Figure 13 — Bottom-track check", _caption_btrack(v)))
        if v.sadcp is not None and len(v.sadcp):
            pages.append((lambda f: sadcp_figure(v, fig=f), f"{station}_sadcp.png",
                          "Figure 9 — Ship-ADCP comparison", _caption_sadcp(v)))

    paths = {}
    # independent high-res PNGs (each plot fn makes its own figure)
    _save_pngs(dh, ctd, velocity, figout, station, paths)

    pdf_path = out / f"{station}_report.pdf"
    with PdfPages(pdf_path) as pdf:
        sc = _scorecard_page(qc, station)
        pdf.savefig(sc); plt.close(sc)
        for draw, _, title, caption in pages:
            pg = _figure_page(draw, title, caption)
            pdf.savefig(pg); plt.close(pg)
    paths["report.pdf"] = str(pdf_path)
    return paths


def _save_pngs(dh, ctd, velocity, out, station, paths):
    import matplotlib.pyplot as plt
    def emit(fig, name):
        paths[name] = str(out / name); plt.close(fig)
    emit(raw_dashboard(dh, savepath=str(out / f"{station}_raw.png")), f"{station}_raw.png")
    if dh.has_up:
        emit(alignment_figure(dh, savepath=str(out / f"{station}_alignment.png")),
             f"{station}_alignment.png")
    if ctd is not None:
        emit(depth_figure(dh, ctd, savepath=str(out / f"{station}_depth.png")),
             f"{station}_depth.png")
        emit(edit_figure(dh, ctd, savepath=str(out / f"{station}_edit.png")),
             f"{station}_edit.png")
    if velocity is not None:
        v = velocity
        emit(velocity_figure(v.vp, bottom=v.bp, station=station,
                             savepath=str(out / f"{station}_velocity.png")),
             f"{station}_velocity.png")
        emit(shear_figure(v.shear, station=station,
                          savepath=str(out / f"{station}_shear.png")),
             f"{station}_shear.png")
        emit(inverse_diagnostics_figure(v, station=station,
                                        savepath=str(out / f"{station}_inverse.png")),
             f"{station}_inverse.png")
        if v.err is not None:
            emit(error_figure(v, station=station,
                              savepath=str(out / f"{station}_error.png")),
                 f"{station}_error.png")
        if v.drift is not None:
            emit(drift_figure(v, station=station,
                              savepath=str(out / f"{station}_drift.png")),
                 f"{station}_drift.png")
        if v.btrk is not None and v.btrk.n_own:
            emit(btrack_figure(v, station=station,
                               savepath=str(out / f"{station}_btrack.png")),
                 f"{station}_btrack.png")
        if v.sadcp is not None and len(v.sadcp):
            emit(sadcp_figure(v, station=station,
                              savepath=str(out / f"{station}_sadcp.png")),
                 f"{station}_sadcp.png")


# ---------------------------------------------------------------- scorecard ---

def _scorecard_page(qc: QCMetrics, station: str):
    import matplotlib.pyplot as plt
    from matplotlib.patches import Circle

    fig = plt.figure(figsize=_A4)
    ax = fig.add_axes([0, 0, 1, 1]); ax.axis("off")

    ax.text(0.5, 0.955, f"LADCP Acquisition QA — {station}", ha="center",
            fontsize=20, fontweight="bold")
    ov = qc.overall_status
    ax.add_patch(Circle((0.17, 0.905), 0.015, color=_COLOR[ov]))
    ax.text(0.205, 0.905, f"overall: {_LABEL[ov]}", va="center", fontsize=15,
            fontweight="bold", color=_COLOR[ov])

    y = 0.855
    y = _section(ax, "Acquisition health", _HEADLINE, qc, y)
    checks = [k for k in _CONSISTENCY if k in qc.metrics]
    if checks:
        y = _section(ax, "Solution consistency (checkinv)", checks, qc, y - 0.02)
    extras = [k for k in qc.metrics if k.startswith("edit_")]
    y = _section(ax, "Editing / screening (informational)", extras, qc, y - 0.02)

    if qc.warnings:
        ax.text(0.07, y - 0.01, "Warnings", fontsize=13, fontweight="bold")
        y -= 0.04
        for w in qc.warnings:
            ax.text(0.10, y, f"• {w.strip()}", fontsize=10.5, color="#b9770e")
            y -= 0.026

    ax.text(0.5, 0.03,
            "● green = OK     ● amber = CHECK     ● red = FAIL\n"
            "Following pages show each figure with notes on what to expect.",
            ha="center", fontsize=10, color="#555", linespacing=1.6)
    return fig


def _section(ax, title, keys, qc, y):
    from matplotlib.patches import Circle
    ax.text(0.07, y, title, fontsize=14, fontweight="bold")
    y -= 0.042
    for k in keys:
        m = qc.metrics.get(k)
        if m is None:
            continue
        ax.add_patch(Circle((0.095, y + 0.006), 0.0095, color=_COLOR[m.status]))
        ax.text(0.13, y, k.replace("_", " "), fontsize=11.5, va="center")
        ax.text(0.55, y, f"{_fmt(m.value)} {m.unit}".strip(), fontsize=11.5, va="center")
        if m.note:
            ax.text(0.13, y - 0.016, m.note[:98], fontsize=8.5, va="center", color="#666")
            y -= 0.015
        y -= 0.03
    return y


def _fmt(v):
    return "[" + ", ".join(str(x) for x in v) + "]" if isinstance(v, list) else str(v)


# ------------------------------------------------------------- figure pages ---

def _figure_page(draw, title, caption):
    """A4 page: title, the figure drawn as vector (top), caption (reserved bottom)."""
    import matplotlib.pyplot as plt

    fig = plt.figure(figsize=_A4, constrained_layout=True)
    fig.suptitle(title, fontsize=16, fontweight="bold")
    top, bot = fig.subfigures(2, 1, height_ratios=[9.0, 2.4])
    draw(top)                                          # vector render into sub-figure
    cax = bot.subplots(); cax.axis("off")
    cax.text(0.01, 0.98, _wrap(caption), va="top", ha="left", fontsize=9.5,
             linespacing=1.5, transform=cax.transAxes,
             bbox=dict(boxstyle="round,pad=0.7", fc="#f4f6f7", ec="#bbb"))
    return fig


def _wrap(text, width=104):
    out = []
    for line in text.split("\n"):
        out.append(textwrap.fill(line, width=width, subsequent_indent="   ")
                   if line.strip() else line)
    return "\n".join(out)


# ---------------------------------------------------------------- captions ----

def _g(qc, k, default="-"):
    m = qc.metrics.get(k)
    return m.value if m else default


def _caption_raw(qc):
    dn, up = _g(qc, "beam_performance_down"), _g(qc, "beam_performance_up")
    rng_up = qc.metrics.get("profiling_range_up")
    sub = (f"  Here the up-looker has {rng_up.note}."
           if rng_up and rng_up.status == Status.WARN else "")
    return (
        "What this shows and what to expect:\n"
        "• W 'bowtie' (top): vertical velocity vs range and time. Expect a clean "
        "descent-then-ascent V in coherent colour; white = no/edited data, salt-and-pepper "
        "speckle = noise.\n"
        f"• Tilt (red): expect mostly below ~10 deg; the dashed line is the 22 deg "
        f"rejection limit (max here {_g(qc,'tilt_max')} deg).\n"
        "• Heading + transmit voltage: expect smooth package rotation and a steady "
        "voltage (a falling voltage means a tiring battery).\n"
        f"• Beam performance: median echo per beam; expect all four within a few % "
        f"(down {dn}, up {up} % of the best beam).\n"
        "• Range of good data: usable range per beam from correlation; a red curve marks "
        f"a beam reaching noticeably less far (sub-nominal).{sub}")


def _caption_align(qc):
    off = _g(qc, "dual_head_offset_est")
    return (
        "What this shows and what to expect:\n"
        "Up-looker minus down-looker heading / pitch / roll, plotted against the "
        "down-looker value.\n"
        f"• Heading difference should form a TIGHT, FLAT band (here about {off} deg) — "
        "that constant offset is just how the two instruments are bolted together, and a "
        "stable band means the compasses are consistent.\n"
        "• Concern if the band is broad or drifts with heading: that points to a "
        "compass / declination problem.\n"
        "• Pitch and roll differences should be small and centred. The exact offset is "
        "refined later at the velocity stage.")


def _caption_depth(qc):
    corr = _g(qc, "ctd_sync_corr")
    bot = qc.metrics.get("bottom_depth")
    ok = isinstance(corr, (int, float)) and corr > 0.9
    verdict = ("good — CTD and LADCP clocks are well aligned" if ok
               else "WEAK — treat depth and bottom with caution")
    return (
        "What this shows and what to expect:\n"
        "• Top — package-depth trajectory with detected water entry/exit lines. Expect a "
        "single clean V down to the turn-around and back to the surface.\n"
        f"• CTD-to-LADCP synchronization correlation = {corr} ({verdict}).\n"
        "• Bottom — per-ping seabed estimates (orange) should cluster tightly on the "
        f"fitted line ({bot.value if bot else '-'} m). A wide spread, or a descent-vs-ascent "
        "split, indicates a rough/sloped seabed or poor synchronization.")


def _caption_edit(qc):
    return (
        "What this shows and what to expect:\n"
        "Target strength (echo) for every bin and ensemble, before and after editing.\n"
        "• Before: the bright diagonal is the approaching/receding seabed echo; the warm "
        "band is the water-column signal.\n"
        "• After: the white wedge near the seabed is removed below-bottom + side-lobe "
        "contamination, and the masked bin-1 rows remove transducer ringing.\n"
        "• Expect the coherent water-column signal to survive and only the contaminated "
        "regions to be blanked. (Edit counts are indicative — see the scorecard.)")


def _caption_velocity(v: VelocityResult):
    vp = v.vp
    bt = (f"  Red/blue dots are the independent bottom-track-referenced velocities "
          f"({v.bp.n_bins} bins); they should sit on the profile near the seabed."
          if v.bp is not None and v.bp.n_bins > 0 else "")
    return (
        "What this shows and what to expect:\n"
        "• The final absolute ocean velocity: east (u, blue) and north (v, red) vs depth, "
        "with the shaded uncertainty band.\n"
        f"• Dotted vertical lines mark the depth-mean (barotropic) reference "
        f"(ū={vp.ubar:+.3f}, v̄={vp.vbar:+.3f} m/s).\n"
        f"• The right panel is the number of velocity samples per bin (range / coverage)."
        f"{bt}\n"
        "• Expect a smooth profile; a flat artificial tail at the seabed would signal "
        "below-bottom contamination (here removed).")


def _caption_shear(v: VelocityResult):
    return (
        "What this shows and what to expect:\n"
        "The shear method is the backbone of the solution — it differences velocity in "
        "depth and integrates, so it is insensitive to the per-cell editing weights.\n"
        "• Left: vertical shear ∂u/∂z, ∂v/∂z per bin. Expect coherent structure, largest "
        "where the flow changes fastest with depth.\n"
        "• Middle: the integrated baroclinic profile (zero-mean by construction) — the "
        "shape of the flow before the barotropic reference is added.\n"
        "• Right: shear samples per bin; thin coverage (few samples) means a noisier "
        "estimate at that depth.")


def _caption_inverse(v: VelocityResult):
    return (
        "What this shows and what to expect:\n"
        "Diagnostics for the reduced shear + reference solution (ps.shear = 1).\n"
        "• Left: the decomposition — dashed = baroclinic shape, solid = absolute solution; "
        "the gap between them is the barotropic reference that the bottom track pins.\n"
        f"• Middle: per-cell fit residual (data minus the shared shear profile), rms "
        f"{v.resid_rms:.3f} m/s. Expect a symmetric cloud centred on zero with no depth "
        "trend.\n"
        "• Right: the residual distribution — a tight, zero-centred peak means the single "
        "baroclinic profile explains the super-ensembles well.")


def _caption_error(v: VelocityResult):
    e = v.err
    return (
        "What this shows and what to expect:\n"
        "Per-cell solution misfit and resolved velocity over the cast's super-ensembles "
        "(each column is one super-ensemble at its package depth, so depth vs super-ensemble "
        "traces the descent/ascent V).\n"
        f"• Left: U/V residual about the shared baroclinic shape — std {e.u_std:.3f} / "
        f"{e.v_std:.3f} m/s. Expect a fine red/blue speckle with no coherent patch; a "
        "structured block flags a bad super-ensemble or depth range.\n"
        "• Middle: median residual per instrument bin # — should hug zero; the far (high-|bin|) "
        "cells are noisiest.\n"
        "• Right: the resolved ocean U/V velocity at each cell.")


def _caption_drift(v: VelocityResult):
    return (
        "What this shows and what to expect:\n"
        "Horizontal tracks during the cast, local metres from the deployment start: the ship "
        "GPS (dots, coloured by speed over ground) and the package dead-reckoned from the "
        "LADCP velocity (blue), with start/bottom/end marked.\n"
        "• On a station-keeping cast the ship stays in a tight cluster; the package loops out "
        "and back as it descends and is advected, closing near the start on recovery.\n"
        "• A package track that does not return, or a wildly large excursion, points to a "
        "reference/dead-reckoning problem.")


def _caption_btrack(v: VelocityResult):
    b = v.btrk
    rdi = f"{b.n_rdi} RDI pings" if b.n_rdi else "no RDI track"
    return (
        "What this shows and what to expect:\n"
        "Two independent bottom-track estimates of the package velocity over ground — our "
        f"own near-seabed-cell track ({b.n_own} pings) and the RDI firmware track "
        f"({rdi}). They reference the LADCP barotropic velocity, so they should agree.\n"
        f"• U/V/W panels: own (blue) over RDI (green); medians marked. Bias u {b.u_bias:+.3f} "
        f"v {b.v_bias:+.3f} m/s — expect |bias| < 0.1; a larger offset flags a bottom-track "
        "or reference problem.\n"
        f"• Bottom roughness {b.roughness:.0f} m (scatter of the detected seabed distance); "
        "large values mean rough or sloped terrain that widens the scatter.")


def _caption_sadcp(v: VelocityResult):
    rms = sadcp_rms_discrepancy(v)
    return (
        "What this shows and what to expect:\n"
        "The shipboard ADCP (SADCP) measures the upper-ocean velocity independently of the "
        "LADCP, so it is both a constraint on the inverse and an external accuracy check.\n"
        "• Left: absolute SADCP U/V (points, with their scatter) over the LADCP inverse "
        "profile (lines). Expect them to track over the SADCP's depth range.\n"
        f"• Right: the LADCP − SADCP difference vs depth, rms {rms:.3f} m/s over the shared "
        "range. Expect a small, depth-uniform offset; a large or sheared difference flags a "
        "reference, timing, or position problem.")
