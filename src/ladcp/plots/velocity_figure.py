"""Velocity profile figure — modern equivalent of LDEO_IX Figure 1.

Panels:
  * east (u) and north (v) ocean velocity vs depth, with the uncertainty band
  * number of velocity samples per depth bin (profiling coverage / range)
"""

from __future__ import annotations

from ..qa.inverse import BottomProfile, VelocityProfile


def velocity_figure(vp: VelocityProfile, *, station: str = "",
                    bottom: BottomProfile | None = None,
                    fig=None, savepath: str | None = None):
    import matplotlib.pyplot as plt

    own = fig is None
    if fig is None:
        fig = plt.figure(figsize=(9, 8), constrained_layout=True)
    axes = fig.subplots(1, 2, width_ratios=[2.2, 1])

    ax = axes[0]
    ax.axvline(0, color="0.7", lw=0.8)
    ax.fill_betweenx(vp.z, vp.u - vp.uerr, vp.u + vp.uerr, color="#2980b9", alpha=0.18)
    ax.fill_betweenx(vp.z, vp.v - vp.uerr, vp.v + vp.uerr, color="#c0392b", alpha=0.18)
    ax.plot(vp.u, vp.z, color="#2980b9", lw=1.6, label="u (east)")
    ax.plot(vp.v, vp.z, color="#c0392b", lw=1.6, label="v (north)")
    if bottom is not None and bottom.n_bins > 0:            # bottom-track cross-check
        ax.plot(bottom.u, bottom.z, color="#2980b9", ls="none", marker=".", ms=4,
                label="u (bottom track)")
        ax.plot(bottom.v, bottom.z, color="#c0392b", ls="none", marker=".", ms=4)
    ax.axvline(vp.ubar, color="#2980b9", ls=":", lw=1.0)
    ax.axvline(vp.vbar, color="#c0392b", ls=":", lw=1.0)
    ax.invert_yaxis()
    ax.set(xlabel="velocity [m/s]", ylabel="depth [m]")
    ax.legend(fontsize=9, loc="lower left")
    ax.set_title(f"barotropic ref: ū={vp.ubar:+.3f}  v̄={vp.vbar:+.3f} m/s",
                 fontsize=9)

    ax = axes[1]
    ax.plot(vp.nvel, vp.z, color="#27ae60", lw=1.4)
    ax.invert_yaxis()
    ax.set(xlabel="samples / bin", ylabel="")
    ax.set_title("coverage", fontsize=9)

    if own:
        fig.suptitle(f"{station} — ocean velocity profile (shear + barotropic)",
                     fontsize=12)
    if savepath:
        fig.savefig(savepath, dpi=200)
    return fig
