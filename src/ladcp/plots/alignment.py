"""Dual-head alignment figure — modern equivalent of LDEO_IX Figure 6.

Shows the up-minus-down attitude differences against the down-looker value:
  * heading difference vs heading       (the sinusoid whose amplitude is the mounting offset)
  * pitch difference vs pitch
  * roll difference vs roll
Titles carry the raw-ping offset estimates (the bit-exact values are velocity-stage).
"""

from __future__ import annotations

from ..qa.attitude import dual_head_offset
from ..qa.ingest import DualHead


def _wrap180(x):
    return (x + 180.0) % 360.0 - 180.0


def alignment_figure(dh: DualHead, *, fig=None, savepath: str | None = None):
    import matplotlib.pyplot as plt

    if not dh.has_up:
        raise ValueError("alignment figure requires an up-looker")

    n = min(dh.down.n_ens, dh.up.n_ens)
    hdg_d, hdg_u = dh.down.heading[:n], dh.up.heading[:n]
    pit_d, pit_u = dh.down.pitch[:n], dh.up.pitch[:n]
    rol_d, rol_u = dh.down.roll[:n], dh.up.roll[:n]
    off, pit_off, rol_off = dual_head_offset(dh)

    own = fig is None
    if own:
        fig = plt.figure(figsize=(8, 10), constrained_layout=True)
    axes = fig.subplots(3, 1)
    axes[0].scatter(hdg_d, _wrap180(hdg_u - hdg_d), s=3, alpha=0.4, color="#2c3e50")
    axes[0].set(xlabel="heading down [deg]", ylabel="heading diff up-down [deg]",
                xlim=(0, 360))
    axes[0].set_title(f"heading offset (tilt-based est): {off:.2f} deg  "
                      f"[velocity-stage value is exact]")

    axes[1].scatter(pit_d, pit_u - pit_d, s=3, alpha=0.4, color="#c0392b")
    axes[1].set(xlabel="pitch down [deg]", ylabel="pitch diff up-down [deg]")
    axes[1].set_title(f"pitch offset est: {pit_off:.2f} deg")

    axes[2].scatter(rol_d, rol_u - rol_d, s=3, alpha=0.4, color="#16a085")
    axes[2].set(xlabel="roll down [deg]", ylabel="roll diff up-down [deg]")
    axes[2].set_title(f"roll offset est: {rol_off:.2f} deg")

    if own:
        fig.suptitle(f"{dh.station} — dual-head alignment", fontweight="bold")
    if savepath:
        fig.savefig(savepath, dpi=200)
    return fig
