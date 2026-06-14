"""Editing before/after figure — modern equivalent of LDEO_IX Figure 14.

Shows the combined up+down target-strength field as bin-number vs ensemble, before and
after the bin-mask / side-lobe / below-bottom edits. The characteristic white wedge near
the seabed in the "after" panel is the removed side-lobe + below-bottom contamination.
"""

from __future__ import annotations

import numpy as np

from ..models import CTDTimeSeries
from ..qa.bottom import detect_bottom
from ..qa.depth import synchronize
from ..qa.edit import edit_data
from ..qa.ingest import DualHead


def edit_figure(dh: DualHead, ctd: CTDTimeSeries, *, fig=None, savepath: str | None = None):
    import matplotlib.pyplot as plt

    sync = synchronize(dh, ctd)
    bottom = detect_bottom(dh, sync,
                           zbottom=getattr(dh.params, "zbottom", float("nan")),
                           guessbottom=getattr(dh.params, "guessbottom", float("nan")))
    er = edit_data(dh, sync, bottom)

    ens = np.arange(er.ts_before.shape[1])
    bn = er.bin_no
    finite = er.ts_before[np.isfinite(er.ts_before)]
    vmin, vmax = np.percentile(finite, [2, 98])

    own = fig is None
    if own:
        fig = plt.figure(figsize=(13, 8), constrained_layout=True)
    axes = fig.subplots(2, 1, sharex=True)
    for ax, field, title in (
        (axes[0], er.ts_before, "Before editing"),
        (axes[1], er.ts_after, "After editing"),
    ):
        im = ax.pcolormesh(ens, bn, field, cmap="turbo", vmin=vmin, vmax=vmax,
                           shading="auto", rasterized=True)
        ax.axhline(0, color="k", lw=0.6)
        ax.set_ylabel("bin #  (up <0, down >0)")
        ax.set_title(title)
        fig.colorbar(im, ax=ax, pad=0.01, label="target strength [dB]")
    axes[1].set_xlabel("ensemble #")
    if own:                                            # standalone PNG only
        c = er.counts
        fig.suptitle(
            f"{dh.station} — data editing   [bin-mask ~{c['bin_mask']}, "
            f"side-lobe ~{c['side_lobe']}, below-bottom ~{c['below_bottom']}]",
            fontweight="bold")
    if savepath:
        fig.savefig(savepath, dpi=200)
    return fig
