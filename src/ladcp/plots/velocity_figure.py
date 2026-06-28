"""Velocity profile figure — modern equivalent of LDEO_IX Figure 1.

Panels:
  * east (u) and north (v) ocean velocity vs depth, with the uncertainty band.
    The x-axis is scaled to the *solution* envelope, not the bottom-track scatter,
    so baroclinic structure stays legible; near-bottom BT points are overlaid as a
    reliability-graded cross-check (solid near the seabed, faded at the noisy far
    edge of the bottom-track range) and BT points beyond the solution scale collapse
    to ``< >`` margin ticks instead of stretching the axis.
  * near-bottom bottom-track cross-check on its OWN x-axis and a depth zoom (only when
    bottom track is present) — read the BT agreement here, without it polluting the
    main scale.
  * number of velocity samples per depth bin (profiling coverage / range)
"""

from __future__ import annotations

import numpy as np

from ..qa.inverse import BottomProfile, VelocityProfile

BLUE = "#2980b9"
RED = "#c0392b"


def _solution_xlim(vp: VelocityProfile, pad: float = 0.04) -> tuple[float, float]:
    """X-limits driven by the solution envelope (u/v +/- uerr), never the BT scatter."""
    lo = np.nanmin([np.nanmin(vp.u - vp.uerr), np.nanmin(vp.v - vp.uerr), 0.0])
    hi = np.nanmax([np.nanmax(vp.u + vp.uerr), np.nanmax(vp.v + vp.uerr), 0.0])
    if not (np.isfinite(lo) and np.isfinite(hi)) or hi - lo < 1e-6:
        return -0.5, 0.5
    return float(lo) - pad, float(hi) + pad


def _bt_reliability(bottom: BottomProfile):
    """Per-bin BT trust 0..1 -> alpha + marker size.

    Dominated by height above the seabed (the far edge of the bottom-track range has
    weak echo and few samples), lightly blended with the formal per-bin error. Returns
    ``(seabed_depth, alpha, size)``.
    """
    seabed = float(np.nanmax(bottom.z))
    hab = seabed - bottom.z                       # metres above bottom
    span = np.nanmax(hab) - np.nanmin(hab)
    rel_h = 1.0 - (hab - np.nanmin(hab)) / (span + 1e-9)
    rel = rel_h
    err = bottom.uerr
    if np.isfinite(err).any() and np.nanmax(err) - np.nanmin(err) > 1e-9:
        rel_e = 1.0 - (err - np.nanmin(err)) / (np.nanmax(err) - np.nanmin(err) + 1e-9)
        rel = 0.65 * rel_h + 0.35 * rel_e
    rel = np.clip(np.nan_to_num(rel, nan=0.0), 0.0, 1.0)
    alpha = np.clip(0.25 + 0.7 * rel, 0.0, 1.0)
    size = 14.0 + 40.0 * rel
    return seabed, alpha, size


def _draw_main(ax, vp: VelocityProfile, bottom: BottomProfile | None):
    ax.axvline(0, color="0.7", lw=0.8)
    ax.fill_betweenx(vp.z, vp.u - vp.uerr, vp.u + vp.uerr, color=BLUE, alpha=0.18)
    ax.fill_betweenx(vp.z, vp.v - vp.uerr, vp.v + vp.uerr, color=RED, alpha=0.18)
    ax.plot(vp.u, vp.z, color=BLUE, lw=1.6, label="u (east)")
    ax.plot(vp.v, vp.z, color=RED, lw=1.6, label="v (north)")
    ax.axvline(vp.ubar, color=BLUE, ls=":", lw=1.0)
    ax.axvline(vp.vbar, color=RED, ls=":", lw=1.0)

    lo, hi = _solution_xlim(vp)
    handles = [ax.lines[1], ax.lines[2]]
    labels = ["u (east)", "v (north)"]
    if bottom is not None and bottom.n_bins > 0:
        import matplotlib.pyplot as plt
        _, alpha, size = _bt_reliability(bottom)
        for val, col in ((bottom.u, BLUE), (bottom.v, RED)):
            ax.scatter(np.clip(val, lo, hi), bottom.z, s=size, c=col,
                       alpha=alpha, edgecolors="none", zorder=2)
            off_lo, off_hi = val < lo, val > hi
            ax.scatter(np.full(int(off_lo.sum()), lo), bottom.z[off_lo], marker="<",
                       s=24, c=col, alpha=0.5, zorder=3, clip_on=False)
            ax.scatter(np.full(int(off_hi.sum()), hi), bottom.z[off_hi], marker=">",
                       s=24, c=col, alpha=0.5, zorder=3, clip_on=False)
        handles += [plt.Line2D([], [], color="0.3", ls="none", marker="o", ms=6),
                    plt.Line2D([], [], color="0.6", ls="none", marker="o", ms=4)]
        labels += ["BT near seabed", "BT far edge"]
    ax.set_xlim(lo, hi)
    ax.invert_yaxis()
    ax.set(xlabel="velocity [m/s]", ylabel="depth [m]")
    ax.legend(handles, labels, fontsize=8, loc="lower left")
    ax.set_title(f"barotropic ref: ū={vp.ubar:+.3f}  v̄={vp.vbar:+.3f} m/s", fontsize=9)


def _draw_bt_panel(ax, vp: VelocityProfile, bottom: BottomProfile):
    """Near-bottom BT cross-check: own auto-scaled x-axis and a depth zoom to the BT span."""
    seabed, alpha, size = _bt_reliability(bottom)
    z_top = float(np.nanmin(bottom.z))
    ax.axvline(0, color="0.7", lw=0.7)
    in_band = (vp.z >= z_top - 40) & (vp.z <= seabed + 40)
    ax.plot(vp.u[in_band], vp.z[in_band], color=BLUE, lw=1.4)
    ax.plot(vp.v[in_band], vp.z[in_band], color=RED, lw=1.4)
    ax.scatter(bottom.u, bottom.z, s=size, c=BLUE, alpha=alpha, edgecolors="none")
    ax.scatter(bottom.v, bottom.z, s=size, c=RED, alpha=alpha, edgecolors="none")
    ax.axhline(seabed, color="0.35", lw=1.3, ls="--")
    ax.text(0.04, 0.04, "seabed", transform=ax.transAxes, fontsize=8, color="0.35")
    ax.set_ylim(seabed + 30, z_top - 20)          # zoomed, depth increasing downward
    ax.set_xlabel("velocity [m/s]")
    ax.set_title("near-bottom BT\ncross-check (own x)", fontsize=9)
    ax.tick_params(labelsize=8)


def velocity_figure(vp: VelocityProfile, *, station: str = "",
                    bottom: BottomProfile | None = None,
                    fig=None, savepath: str | None = None):
    import matplotlib.pyplot as plt

    own = fig is None
    if fig is None:
        fig = plt.figure(figsize=(9, 8), constrained_layout=True)

    has_bt = bottom is not None and bottom.n_bins > 0
    if has_bt:
        axes = fig.subplots(1, 3, width_ratios=[2.3, 1.1, 0.9])
        ax_main, ax_bt, ax_cov = axes
    else:
        ax_main, ax_cov = fig.subplots(1, 2, width_ratios=[2.2, 1])
        ax_bt = None

    _draw_main(ax_main, vp, bottom)
    if ax_bt is not None:
        _draw_bt_panel(ax_bt, vp, bottom)

    ax_cov.plot(vp.nvel, vp.z, color="#27ae60", lw=1.4)
    ax_cov.invert_yaxis()
    ax_cov.set(xlabel="samples / bin", ylabel="")
    ax_cov.set_title("coverage", fontsize=9)

    if own:
        fig.suptitle(f"{station} — ocean velocity profile (shear + barotropic)",
                     fontsize=12)
    if savepath:
        fig.savefig(savepath, dpi=200)
    return fig
