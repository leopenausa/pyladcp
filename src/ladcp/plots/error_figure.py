"""Super-ensemble error figure — LDEO_IX Figure 3 (``geterr.m``).

Images the per-cell solution misfit and the resolved velocity over the cast's
super-ensembles. Because each super-ensemble holds the bins at its package depth, plotting
depth against super-ensemble number traces the descent/ascent "V".

Two rows (U, then V), three columns:

  1. residual field — depth vs super-ensemble #, coloured by the per-cell residual about the
     shared baroclinic shape (title carries the outlier-resistant std);
  2. median residual per instrument bin # — exposes range-dependent bias (far bins noisier);
  3. ocean-velocity field — depth vs super-ensemble #, coloured by the solution velocity at
     each cell.
"""

from __future__ import annotations

import numpy as np

from ..qa.inverse import VelocityResult

_CMAP = "seismic"       # blue=negative, white=0, red=positive -- matches legacy geterr.m


def _clim(a, pct=98.0):
    a = np.abs(a[np.isfinite(a)])
    return float(np.nanpercentile(a, pct)) if a.size else 1.0


def error_figure(r: VelocityResult, *, station: str = "", fig=None,
                 savepath: str | None = None):
    """Figure 3 — super-ensemble residual + velocity field. Requires ``r.err``."""
    import matplotlib.pyplot as plt

    own = fig is None
    if fig is None:
        fig = plt.figure(figsize=(12, 7), constrained_layout=True)
    axes = fig.subplots(2, 3, width_ratios=[1.3, 1.0, 1.3])
    e = r.err
    if e is None:
        axes[0, 0].text(0.5, 0.5, "no error field", ha="center", va="center",
                        transform=axes[0, 0].transAxes, color="0.5")
        if savepath:
            fig.savefig(savepath, dpi=200)
        return fig

    # scale the residual colour to the robust spread (~2.5 sigma) so the field structure
    # shows through instead of being washed out by a few heavy-tailed outliers
    rlim = 2.5 * max(s for s in (e.u_std, e.v_std) if np.isfinite(s))

    rows = [("U", e.resid_u, e.u_oce, e.u_std), ("V", e.resid_v, e.v_oce, e.v_std)]
    for i, (name, resid, oce, std) in enumerate(rows):
        fin = np.isfinite(resid) & np.isfinite(e.depth)
        # 1 - residual field: depth vs super-ensemble number
        ax = axes[i, 0]
        sc = ax.scatter(e.se_index[fin], e.depth[fin], c=resid[fin], s=4,
                        cmap=_CMAP, vmin=-rlim, vmax=rlim, marker="s", linewidths=0)
        ax.invert_yaxis()
        ax.set(xlabel="super ensemble #", ylabel="depth [m]")
        ax.set_title(f"{name}-err std: {std:.3f}", fontsize=10)
        fig.colorbar(sc, ax=ax, fraction=0.05, pad=0.02)

        # 2 - median residual per instrument bin number
        ax = axes[i, 1]
        with np.errstate(invalid="ignore"):
            med = np.array([np.nanmedian(resid[k]) if np.isfinite(resid[k]).any() else np.nan
                            for k in range(resid.shape[0])])
        order = np.argsort(e.binno)
        ax.plot(med[order], e.binno[order], color="#2980b9", lw=1.0)
        ax.axvline(0, color="0.7", lw=0.8)
        ax.set(xlabel="residual [m/s]", ylabel="bin #")
        ax.set_xlim(-0.05, 0.05)
        ax.set_title(f"median({name}-err)", fontsize=10)

        # 3 - ocean-velocity field: depth vs super-ensemble number
        ax = axes[i, 2]
        olim = _clim(oce)
        fo = np.isfinite(oce) & np.isfinite(e.depth)
        sc = ax.scatter(e.se_index[fo], e.depth[fo], c=oce[fo], s=4, cmap=_CMAP,
                        vmin=-olim, vmax=olim, marker="s", linewidths=0)
        ax.invert_yaxis()
        ax.set(xlabel="ensemble #", ylabel="depth [m]")
        ax.set_title(f"{name}$_{{oce}}$", fontsize=10)
        fig.colorbar(sc, ax=ax, fraction=0.05, pad=0.02)

    if own:
        fig.suptitle(f"{station} — super-ensemble error (Figure 3)", fontsize=12)
    if savepath:
        fig.savefig(savepath, dpi=200)
    return fig
