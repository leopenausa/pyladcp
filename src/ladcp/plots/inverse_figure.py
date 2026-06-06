"""Inversion-diagnostics figure — modern equivalent of LDEO_IX Figure 12.

Legacy Figure 12 plots the constraint weights of the full sparse inverse. We use the
reduced shear + reference solution (``ps.shear==1``), so the honest diagnostics are:

  * the **decomposition** — baroclinic shear shape vs the absolute solution after the
    barotropic (depth-mean) reference is added;
  * the **fit residual** — how well the shared baroclinic profile explains each
    super-ensemble cell (data minus shear fit), versus depth;
  * the **residual distribution** — should be tight and centred on zero.
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
