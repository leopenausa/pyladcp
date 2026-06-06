"""Shear-profile figure — modern equivalent of LDEO_IX Figure 3.

The shear method is the backbone of the LADCP solution: it forms the vertical shear of
horizontal velocity ``d(u)/dz``, ``d(v)/dz`` in each depth bin and integrates it to the
baroclinic profile. This figure shows the binned shear, the integrated baroclinic profile,
and the number of shear samples per bin (the statistical backbone of each estimate).
"""

from __future__ import annotations

import numpy as np

from ..qa.inverse import ShearProfile


def shear_figure(sp: ShearProfile, *, station: str = "",
                 fig=None, savepath: str | None = None):
    import matplotlib.pyplot as plt

    own = fig is None
    if fig is None:
        fig = plt.figure(figsize=(9, 8), constrained_layout=True)
    axes = fig.subplots(1, 3, width_ratios=[1.6, 1.6, 1])

    us, vs = sp.u_shear * 1e3, sp.v_shear * 1e3            # 1/s -> 1e-3 1/s

    ax = axes[0]
    ax.axvline(0, color="0.7", lw=0.8)
    ax.plot(us, sp.z, color="#2980b9", lw=1.4, label="∂u/∂z")
    ax.plot(vs, sp.z, color="#c0392b", lw=1.4, label="∂v/∂z")
    ax.invert_yaxis()
    ax.set(xlabel="shear [10⁻³ s⁻¹]", ylabel="depth [m]")
    ax.legend(fontsize=9, loc="lower left")
    ax.set_title("vertical shear", fontsize=9)

    ax = axes[1]
    ax.axvline(0, color="0.7", lw=0.8)
    ax.plot(sp.u, sp.z, color="#2980b9", lw=1.6, label="u (east)")
    ax.plot(sp.v, sp.z, color="#c0392b", lw=1.6, label="v (north)")
    ax.invert_yaxis()
    ax.set(xlabel="baroclinic velocity [m/s]")
    ax.legend(fontsize=9, loc="lower left")
    ax.set_title("integrated shear (zero-mean)", fontsize=9)

    ax = axes[2]
    ax.plot(sp.n, sp.z, color="#27ae60", lw=1.4)
    ax.invert_yaxis()
    ax.set(xlabel="shear samples / bin")
    ax.set_title("coverage", fontsize=9)

    if own:
        fig.suptitle(f"{station} — vertical shear profile", fontsize=12)
    if savepath:
        fig.savefig(savepath, dpi=200)
    return fig
