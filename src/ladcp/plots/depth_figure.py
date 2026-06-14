"""Surface & seabed detection figure — modern equivalent of LDEO_IX Figure 4.

Panels:
  * package depth vs ensemble, with water-entry / water-exit marked (surface detection)
  * near-seabed detail: package depth, per-ping seabed depth (z + bottom distance), and
    the fitted seabed line (bottom detection).
"""

from __future__ import annotations

import numpy as np

from ..models import CTDTimeSeries
from ..qa.bottom import detect_bottom
from ..qa.depth import synchronize, water_window
from ..qa.ingest import DualHead


def depth_figure(dh: DualHead, ctd: CTDTimeSeries, *, fig=None, savepath: str | None = None):
    import matplotlib.pyplot as plt

    sync = synchronize(dh, ctd)
    bottom = detect_bottom(dh, sync,
                           zbottom=getattr(dh.params, "zbottom", float("nan")),
                           guessbottom=getattr(dh.params, "guessbottom", float("nan")))
    z = sync.z_on_ping
    ens = np.arange(z.size)
    i0, i1 = water_window(z)
    hbot = bottom.hbot
    seabed = z + hbot                                  # per-ping seabed depth estimate

    if fig is None:
        fig = plt.figure(figsize=(11, 8), constrained_layout=True)
    axes = fig.subplots(2, 1)

    ax = axes[0]
    ax.plot(ens, z, lw=0.8, color="#2c3e50")
    ax.axvline(i0, ls="--", color="#27ae60", label=f"water entry (ens {i0})")
    ax.axvline(i1, ls="--", color="#c0392b", label=f"water exit (ens {i1})")
    ax.axhline(sync.maxdepth, ls=":", color="k", lw=0.8,
               label=f"max depth {sync.maxdepth:.0f} m")
    ax.invert_yaxis()
    ax.set(xlabel="ensemble", ylabel="package depth [m]")
    ax.set_title(f"{dh.station} — surface detection (sync corr {sync.corr:.3f})")
    ax.legend(fontsize=8)

    ax = axes[1]
    near = np.isfinite(seabed)
    ax.plot(ens[near], seabed[near], ".", ms=2, color="#e67e22",
            label="per-ping seabed (z + bottom dist)")
    ax.plot(ens, z, lw=0.8, color="#2c3e50", label="package depth")
    if np.isfinite(bottom.zbottom):
        tag = " — near-bottom lower bound" if bottom.is_fallback else ""
        ax.axhline(bottom.zbottom, ls="--", color="k",
                   label=f"seabed {bottom.zbottom:.1f} +/- {bottom.error:.1f} m{tag}")
    ax.invert_yaxis()
    ax.set(xlabel="ensemble", ylabel="depth [m]")
    if near.any():
        # fall back to the deepest per-ping seabed when bottom detection returned NaN
        ref = (bottom.zbottom if np.isfinite(bottom.zbottom)
               else float(np.nanmax(seabed[near])))
        ax.set_ylim(ref + 40, max(0, ref - 250))
    ax.set_title("seabed detection")
    ax.legend(fontsize=8)

    if savepath:
        fig.savefig(savepath, dpi=200)
    return fig
