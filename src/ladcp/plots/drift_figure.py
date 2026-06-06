"""Ship + CTD drift map — LDEO_IX navigation/drift figure.

The package (CTD) is dead-reckoned from the LADCP-derived package velocity and the ship is
the merged GPS track; both are local east/north metres from the deployment start. On a
station-keeping cast both stay within tens of metres, and the gap between them is the
package's horizontal excursion below the ship.
"""

from __future__ import annotations

import numpy as np

from ..qa.inverse import VelocityResult


def drift_figure(r: VelocityResult, *, station: str = "", fig=None,
                 savepath: str | None = None):
    """Ship (GPS, coloured by speed) + dead-reckoned package track. Requires ``r.drift``."""
    import matplotlib.pyplot as plt

    own = fig is None
    if fig is None:
        fig = plt.figure(figsize=(8, 7), constrained_layout=True)
    ax = fig.subplots(1, 1)
    d = r.drift
    if d is None:
        ax.text(0.5, 0.5, "no navigation", ha="center", va="center",
                transform=ax.transAxes, color="0.5")
        if savepath:
            fig.savefig(savepath, dpi=200)
        return fig

    # package (CTD) dead-reckoned track
    ax.plot(d.pkg_e, d.pkg_n, "-", color="#2980b9", lw=1.3, label="CTD (dead-reckoned)")
    # ship GPS track, coloured by speed over ground
    sog = d.ship_sog
    finite = np.isfinite(d.ship_e) & np.isfinite(d.ship_n)
    ax.plot(d.ship_e[finite], d.ship_n[finite], "-", color="0.7", lw=0.6, zorder=1)
    # clip the colour to a robust max so 1 Hz GPS-jitter spikes don't wash out the scale
    vmax = np.nanpercentile(sog[finite], 95) if finite.any() else 1.0
    sc = ax.scatter(d.ship_e[finite], d.ship_n[finite], c=sog[finite], s=8, cmap="viridis",
                    vmin=0, vmax=max(vmax, 0.05), zorder=2, label="ship (GPS)")
    fig.colorbar(sc, ax=ax, fraction=0.05, pad=0.02, label="ship speed over ground [m/s]")

    # start / bottom / end markers
    if d.pkg_e.size:
        ax.plot(d.pkg_e[0], d.pkg_n[0], "k^", ms=10, label="start")
        ax.plot(d.pkg_e[-1], d.pkg_n[-1], "ks", ms=9, label="end")
        ib = min(d.i_bottom, d.pkg_e.size - 1)
        ax.plot(d.pkg_e[ib], d.pkg_n[ib], "ko", ms=9, mfc="none", label="bottom")

    ax.set(xlabel="east–west [m]", ylabel="north–south [m]")
    ax.set_aspect("equal", adjustable="datalim")
    ax.grid(True, color="0.9")
    ax.legend(fontsize=8, loc="best")
    if own:
        ax.set_title(f"{station} — ship & CTD drift", fontsize=12)
    if savepath:
        fig.savefig(savepath, dpi=200)
    return fig
