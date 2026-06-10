"""Inversion-diagnostics figures (LDEO_IX Figure 12).

Two figures share this module:

* :func:`inverse_diagnostics_figure` — solver-agnostic solution diagnostics:
  the **decomposition** (baroclinic shape vs absolute solution), the **fit residual**
  (data minus the shared baroclinic profile, vs depth), and the **residual
  distribution** (should be tight and centred on zero).
* :func:`constraint_weights_figure` — the literal legacy Figure 12: per-constraint
  accumulated weights of the full inverse (data / smoothing / bottom track /
  ship ADCP / GPS), stacked over the ocean bins and the CTD super-ensembles.
"""

from __future__ import annotations

import numpy as np

from ..qa.inverse import VelocityResult


def inverse_diagnostics_figure(r: VelocityResult, *, station: str = "",
                               fig=None, savepath: str | None = None):
    import matplotlib.pyplot as plt

    own = fig is None
    if fig is None:
        fig = plt.figure(figsize=(9, 8), constrained_layout=True)
    axes = fig.subplots(1, 3, width_ratios=[1.4, 1.4, 1])

    vp, sp = r.vp, r.shear

    # 1 - decomposition: baroclinic shape (dashed) -> absolute solution (solid)
    ax = axes[0]
    ax.axvline(0, color="0.7", lw=0.8)
    ax.plot(sp.u, sp.z, color="#2980b9", lw=1.0, ls="--", label="u baroclinic")
    ax.plot(sp.v, sp.z, color="#c0392b", lw=1.0, ls="--", label="v baroclinic")
    ax.plot(vp.u, vp.z, color="#2980b9", lw=1.7, label="u absolute")
    ax.plot(vp.v, vp.z, color="#c0392b", lw=1.7, label="v absolute")
    ax.invert_yaxis()
    ax.set(xlabel="velocity [m/s]", ylabel="depth [m]")
    ax.legend(fontsize=8, loc="lower left")
    ax.set_title(f"barotropic ref ū={vp.ubar:+.3f} v̄={vp.vbar:+.3f}", fontsize=9)

    # 2 - per-cell fit residual vs depth
    ax = axes[1]
    ax.axvline(0, color="0.7", lw=0.8)
    ax.plot(r.resid_u, r.resid_z, ".", color="#2980b9", ms=2.0, alpha=0.25)
    ax.plot(r.resid_v, r.resid_z, ".", color="#c0392b", ms=2.0, alpha=0.25)
    ax.invert_yaxis()
    ax.set(xlabel="cell − shear fit [m/s]")
    ax.set_xlim(-0.3, 0.3)
    ax.set_title(f"fit residual (rms {r.resid_rms:.3f} m/s)", fontsize=9)

    # 3 - residual distribution
    ax = axes[2]
    bins = np.linspace(-0.3, 0.3, 41)
    ax.hist(r.resid_u[np.isfinite(r.resid_u)], bins=bins, color="#2980b9", alpha=0.5,
            orientation="horizontal", label="u")
    ax.hist(r.resid_v[np.isfinite(r.resid_v)], bins=bins, color="#c0392b", alpha=0.5,
            orientation="horizontal", label="v")
    ax.axhline(0, color="0.7", lw=0.8)
    ax.set(xlabel="count", ylabel="residual [m/s]")
    ax.legend(fontsize=8)
    ax.set_title("distribution", fontsize=9)

    if own:
        fig.suptitle(f"{station} — inversion diagnostics", fontsize=12)
    if savepath:
        fig.savefig(savepath, dpi=200)
    return fig


# stable colour per constraint (legacy Figure 12 palette order)
_CONSTRAINT_COLORS = {
    "velocity": "#1f77b4",
    "smoothing": "#d62728",
    "bottom track": "#e8b33a",
    "ship ADCP": "#7b2d8b",
    "GPS navigation": "#2ca02c",
    "zero-mean": "#7f7f7f",
}


def constraint_weights_figure(w, *, station: str = "", fig=None,
                              savepath: str | None = None):
    """Constraint-weights stacked bars (legacy LDEO_IX Figure 12, full inverse only).

    Top: per-ocean-bin accumulated weight ``sum|w|`` vs depth — how strongly each
    constraint (data, smoothing, bottom track, ship ADCP, GPS) pins the ocean velocity.
    Bottom: the same over the reference/CTD unknowns vs super-ensemble index. ``w`` is the
    :class:`~ladcp.qa.inverse_full.ConstraintWeights` carried on a ``VelocityResult``.
    """
    import matplotlib.pyplot as plt

    own = fig is None
    if fig is None:
        fig = plt.figure(figsize=(9, 8), constrained_layout=True)
    ax_o, ax_c = fig.subplots(2, 1)

    def stacked(ax, x, sums: dict, width):
        bottom = np.zeros_like(x, dtype=float)
        for name, vals in sums.items():
            vals = np.asarray(vals, dtype=float)
            if not np.any(vals > 0):
                continue
            ax.bar(x, vals, width=width, bottom=bottom, label=name,
                   color=_CONSTRAINT_COLORS.get(name, "0.5"))
            bottom += vals
        ax.legend(fontsize=8)

    dz = float(np.median(np.diff(w.z))) if w.z.size > 1 else 8.0
    stacked(ax_o, w.z, w.ocean, width=0.9 * dz)
    ax_o.set(xlabel="depth [m]", ylabel="sum of weights")
    ax_o.set_title("ocean velocity constraints", fontsize=9)

    nt = next(iter(w.ctd.values())).size if w.ctd else 0
    stacked(ax_c, np.arange(1, nt + 1), w.ctd, width=0.9)
    ax_c.set(xlabel="super ensemble", ylabel="sum of weights")
    ax_c.set_title("CTD velocity constraints", fontsize=9)

    if own:
        fig.suptitle(f"{station} — inverse constraint weights", fontsize=12)
    if savepath:
        fig.savefig(savepath, dpi=200)
    return fig
