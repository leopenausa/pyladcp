"""Package dead-reckoning + ship-GPS map — LDEO_IX navigation/drift figure.

The big track is the *package* (CTD+LADCP) dead-reckoned from the LADCP-derived package
velocity: the instrument hanging on the wire, advected by sub-surface currents. The ship
is the merged GPS track; on a DP / station-keeping cast it stays within a few metres while
the package wanders hundreds of metres below it. Both are local east/north metres from the
deployment start. The package excursion is a diagnostic only — it is *not* the velocity
correction; the correction is the GPS-derived barotropic (depth-mean u/v), reported in the
annotation box.
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

    # package (CTD+LADCP) dead-reckoned track — the instrument on the wire, not the ship
    ax.plot(d.pkg_e, d.pkg_n, "-", color="#2980b9", lw=1.3,
            label="package (dead-reckoned, on wire)")
    # ship GPS track, coloured by speed over ground
    sog = d.ship_sog
    finite = np.isfinite(d.ship_e) & np.isfinite(d.ship_n)
    ax.plot(d.ship_e[finite], d.ship_n[finite], "-", color="0.7", lw=0.6, zorder=1)
    # clip the colour to a robust max so 1 Hz GPS-jitter spikes don't wash out the scale
    vmax = np.nanpercentile(sog[finite], 95) if finite.any() else 1.0
    sc = ax.scatter(d.ship_e[finite], d.ship_n[finite], c=sog[finite], s=8, cmap="viridis",
                    vmin=0, vmax=max(vmax, 0.05), zorder=2, label="ship (GPS, DP-held)")
    fig.colorbar(sc, ax=ax, fraction=0.05, pad=0.02, label="ship speed over ground [m/s]")

    # start / bottom / end markers
    if d.pkg_e.size:
        ax.plot(d.pkg_e[0], d.pkg_n[0], "k^", ms=10, label="start")
        ax.plot(d.pkg_e[-1], d.pkg_n[-1], "ks", ms=9, label="end")
        ib = min(d.i_bottom, d.pkg_e.size - 1)
        ax.plot(d.pkg_e[ib], d.pkg_n[ib], "ko", ms=9, mfc="none", label="bottom")

    # annotation: the numbers that actually matter — tiny ship drift, the package
    # excursion (diagnostic), and the GPS barotropic that *is* the velocity correction.
    if finite.any():
        ship_net = float(np.hypot(d.ship_e[finite][-1] - d.ship_e[finite][0],
                                  d.ship_n[finite][-1] - d.ship_n[finite][0]))
    else:
        ship_net = float("nan")
    pkg_max = float(np.nanmax(np.hypot(d.pkg_e, d.pkg_n))) if d.pkg_e.size else float("nan")
    ubar = float(np.nanmean(r.vp.u)) if r.vp.u.size else float("nan")
    vbar = float(np.nanmean(r.vp.v)) if r.vp.v.size else float("nan")
    note = (f"ship net drift: {ship_net:.0f} m\n"
            f"package excursion: {pkg_max:.0f} m (diagnostic)\n"
            f"barotropic correction: u {ubar:+.3f}, v {vbar:+.3f} m/s")
    ax.text(0.02, 0.02, note, transform=ax.transAxes, fontsize=8, va="bottom", ha="left",
            family="monospace", bbox=dict(boxstyle="round", fc="white", ec="0.7", alpha=0.85))

    ax.set(xlabel="east–west [m]", ylabel="north–south [m]")
    ax.set_aspect("equal", adjustable="datalim")
    ax.grid(True, color="0.9")
    ax.legend(fontsize=8, loc="upper right")
    if own:
        ax.set_title(f"{station} — package dead-reckoning (ship DP-held)", fontsize=12)
    if savepath:
        fig.savefig(savepath, dpi=200)
    return fig
