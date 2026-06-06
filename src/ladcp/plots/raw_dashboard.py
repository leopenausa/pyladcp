"""Raw-data dashboard — modern equivalent of LDEO_IX Figure 2.

Panels (same information as the MATLAB figure, re-laid-out for readability):
  * vertical-velocity field W(range, ensemble)        — the "bowtie"
  * tilt / heading / transmit-voltage time series
  * beam performance: median echo amplitude vs distance (4 beams, up + down)
  * range of good data: median correlation vs distance (4 beams, up + down)
"""

from __future__ import annotations

import numpy as np

from ..qa.beams import beam_health
from ..qa.ingest import DualHead
from ..qa.range import profiling_range
from ..qa.screen import tilt_series

_W_COMP = 2          # earth coord component 3 = vertical velocity


def raw_dashboard(dh: DualHead, *, fig=None, savepath: str | None = None):
    """Render the raw-data dashboard. Draws into ``fig`` if given (Figure/SubFigure),
    else creates its own figure (for standalone PNG output)."""
    import matplotlib.pyplot as plt

    d = dh.down
    if fig is None:
        fig = plt.figure(figsize=(13, 11), constrained_layout=True)
    gs = fig.add_gridspec(4, 2, height_ratios=[2.2, 0.9, 0.9, 1.8])

    # --- W field (bowtie), spanning the top row ---
    ax_w = fig.add_subplot(gs[0, :])
    _plot_wfield(ax_w, dh)
    ax_w.set_title(f"{dh.station}  —  vertical velocity W (range vs ensemble)")

    # --- time series: tilt / heading / xmv ---
    tilt, _ = tilt_series(d)
    ens = np.arange(d.n_ens)
    ax_t = fig.add_subplot(gs[1, :])
    ax_t.plot(ens, tilt, lw=0.6, color="#c0392b")
    ax_t.set_ylabel("tilt [deg]"); ax_t.set_ylim(0, max(30, np.nanmax(tilt) * 1.1))
    ax_t.axhline(dh.params.tiltmax[0] if dh.params else 22, ls="--", lw=0.8, color="k")
    ax_t.set_xticklabels([])

    ax_h = fig.add_subplot(gs[2, :])
    ax_h.plot(ens, d.heading, lw=0.5, color="#2c3e50")
    ax_h.set_ylabel("heading [deg]"); ax_h.set_ylim(0, 360)
    ax_h.set_yticks([0, 90, 180, 270, 360])
    ax2 = ax_h.twinx()
    if d.xmit_voltage is not None:
        ax2.plot(ens, d.xmit_voltage, lw=0.5, color="#16a085")
        ax2.set_ylabel("xmit volt", color="#16a085")
    ax_h.set_xlabel("ensemble")

    # --- beam performance + range of good data ---
    _plot_beam_perf(fig.add_subplot(gs[3, 0]), dh)
    _plot_range(fig.add_subplot(gs[3, 1]), dh)

    if savepath:
        fig.savefig(savepath, dpi=200)
    return fig


def _signed_depth(head, sign):
    dist0 = head.meta.get("dist_first_m", head.blank_m + head.cell_m / 2.0)
    return sign * (dist0 + np.arange(head.n_cells) * head.cell_m)


def _plot_wfield(ax, dh):
    # vertical velocity is dominated by the package fall/rise speed (~1.5 m/s); scale to
    # a robust 98th percentile so the coherent bowtie shows instead of saturating.
    allw = np.concatenate([h.vel[_W_COMP].ravel() for h in (dh.down, dh.up) if h is not None])
    vmax = float(np.nanpercentile(np.abs(allw), 98)) or 1.5
    cmap = "RdYlBu_r"
    for head, sign in ((dh.down, 1), (dh.up, -1)) if dh.has_up else ((dh.down, 1),):
        if head is None:
            continue
        w = head.vel[_W_COMP]                       # [ncell, nens]
        z = _signed_depth(head, sign)
        ens = np.arange(head.n_ens)
        pc = ax.pcolormesh(ens, z, w, cmap=cmap, vmin=-vmax, vmax=vmax,
                           shading="auto", rasterized=True)
    ax.axhline(0, color="k", lw=0.8)
    ax.invert_yaxis()
    ax.set_ylabel("range [m]  (down +, up -)")
    ax.set_xlabel("ensemble")
    cb = ax.figure.colorbar(pc, ax=ax, pad=0.01)
    cb.set_label("W [m/s]")


def _plot_beam_perf(ax, dh):
    heads = ((dh.down, 1, "-"), (dh.up, -1, "--")) if dh.has_up else ((dh.down, 1, "-"),)
    for head, sign, style in heads:
        if head is None:
            continue
        ea = np.nanmedian(head.echo, axis=2)        # [4, nbin]
        z = _signed_depth(head, sign)
        bh = beam_health(head)
        for b in range(4):
            ax.plot(ea[b], z, style, lw=0.9, label=f"{head.facing} b{b+1} {bh.pct[b]}%")
    ax.axhline(0, color="k", lw=0.6)
    ax.invert_yaxis()
    ax.set_xlabel("median echo amplitude [counts]")
    ax.set_ylabel("distance [m]")
    ax.set_title("beam performance")
    ax.legend(fontsize=6, ncol=2, loc="best")


def _plot_range(ax, dh):
    for head, sign in ((dh.down, 1), (dh.up, -1)) if dh.has_up else ((dh.down, 1),):
        if head is None:
            continue
        cm = np.nanmedian(head.corr, axis=2)
        z = _signed_depth(head, sign)
        rr = profiling_range(head)
        for b in range(4):
            col = "#c0392b" if rr.flags[b] != "ok" else "#2980b9"
            ax.plot(cm[b], z, lw=0.9, color=col)
        for b in range(4):
            ax.scatter([np.nanmax(cm) * 0.3], [sign * rr.beam_range[b]],
                       s=14, color="k", zorder=3)
    ax.axhline(0, color="k", lw=0.6)
    ax.invert_yaxis()
    ax.set_xlabel("median correlation [counts]")
    ax.set_ylabel("distance [m]")
    ax.set_title("range of good data (red = sub-nominal beam)")
