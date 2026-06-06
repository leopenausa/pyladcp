"""Bottom-track check figure — modern equivalent of LDEO_IX Figure 13 (``checkbtrk``).

``checkbtrk`` compares two independent bottom-track estimates of the package velocity over
ground: the RDI firmware track and our own near-seabed-cell track. Agreement between them
(small inter-method bias, comparable scatter) is the headline bottom-track quality check —
the reference that pins the LADCP barotropic velocity.

The horizontal components (U, V) reference the LADCP barotropic velocity, so the first two
panels overlay the own (blue) and RDI (green) earth-frame bottom-track velocity with their
medians marked and the inter-method bias / own scatter annotated. The third panel shows the
detected seabed-distance distribution, whose spread is the bottom roughness. (Vertical W is
dominated by the package descent/ascent and is not a useful reference check, so it is
summarised numerically on the result rather than plotted.)
"""

from __future__ import annotations

import numpy as np

from ..qa.inverse import VelocityResult

_OWN = "#2980b9"
_RDI = "#27ae60"


def btrack_figure(r: VelocityResult, *, station: str = "", fig=None,
                  savepath: str | None = None):
    """Own vs RDI bottom-track comparison. Requires ``r.btrk``."""
    import matplotlib.pyplot as plt

    own = fig is None
    if fig is None:
        fig = plt.figure(figsize=(9, 7), constrained_layout=True)
    axes = fig.subplots(1, 3)
    b = r.btrk
    if b is None or b.n_own == 0:
        axes[0].text(0.5, 0.5, "no bottom track", ha="center", va="center",
                     transform=axes[0].transAxes, color="0.5")
        if savepath:
            fig.savefig(savepath, dpi=200)
        return fig

    panels = [("U", b.own_u, b.rdi_u, b.u_bias, b.u_std),
              ("V", b.own_v, b.rdi_v, b.v_bias, b.v_std)]
    for ax, (name, ov, rv, bias, std) in zip(axes[:2], panels, strict=True):
        bins = np.linspace(-0.5, 0.5, 41)
        ax.hist(ov, bins=bins, color=_OWN, alpha=0.55, label=f"own (n={ov.size})")
        if rv.size:
            ax.hist(rv, bins=bins, color=_RDI, alpha=0.45, label=f"RDI (n={rv.size})")
        ax.axvline(0, color="0.7", lw=0.8)
        if ov.size:
            ax.axvline(np.median(ov), color=_OWN, lw=1.5)
        if rv.size:
            ax.axvline(np.median(rv), color=_RDI, lw=1.5)
        ax.set(xlabel=f"{name} bottom-track [m/s]")
        bias_txt = f"bias {bias:+.3f}" if np.isfinite(bias) else "bias n/a"
        ax.set_title(f"{name}: {bias_txt}, std {std:.3f}", fontsize=9)
        ax.legend(fontsize=8, loc="upper right")
    axes[0].set_ylabel("count")

    # third panel: detected seabed distance -> spread is the bottom roughness
    ax = axes[2]
    if b.hbot.size:
        ax.hist(b.hbot, bins=30, color="#8e44ad", alpha=0.6)
        ax.axvline(np.median(b.hbot), color="#8e44ad", lw=1.5)
    ax.set(xlabel="seabed distance [m]")
    rough = f"{b.roughness:.0f} m" if np.isfinite(b.roughness) else "n/a"
    ax.set_title(f"bottom roughness {rough}", fontsize=9)

    if own:
        rough = f"{b.roughness:.0f} m" if np.isfinite(b.roughness) else "n/a"
        fig.suptitle(f"{station} — bottom-track check (roughness {rough})", fontsize=12)
    if savepath:
        fig.savefig(savepath, dpi=200)
    return fig
