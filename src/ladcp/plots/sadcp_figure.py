"""Ship-ADCP comparison figure — modern equivalent of LDEO_IX Figure 9.

Legacy ``loadsadcp.m`` Figure 9 plots the shipboard-ADCP (SADCP) U/V profiles used as the
inverse constraint, alongside the ship-nav/SADCP positions. The scientifically useful view
is the **cross-instrument comparison**: the absolute SADCP profile vs the LADCP inverse
solution over their shared depth range, and their difference — the SADCP being an
independent measurement of the same upper-ocean velocity, this is the headline accuracy
check (``sadcp_rms_discrepancy``).

  * left — U and V: SADCP (points, with its scatter ``verr``) over the LADCP inverse (lines);
  * right — LADCP − SADCP vs depth, annotated with the upper-ocean RMS discrepancy.
"""

from __future__ import annotations

import numpy as np

from ..qa.inverse import VelocityResult

_BLUE = "#2980b9"      # east (u)
_RED = "#c0392b"       # north (v)


def sadcp_figure(r: VelocityResult, *, station: str = "", fig=None,
                 savepath: str | None = None):
    """SADCP-vs-LADCP comparison. Requires ``r.sadcp`` (the constraint profile)."""
    import matplotlib.pyplot as plt

    own = fig is None
    if fig is None:
        fig = plt.figure(figsize=(9, 8), constrained_layout=True)
    axes = fig.subplots(1, 2, width_ratios=[1.5, 1])

    vp = r.vp
    sv = r.sadcp
    if sv is None or len(sv) == 0:
        axes[0].text(0.5, 0.5, "no ship-ADCP profile", ha="center", va="center",
                     transform=axes[0].transAxes, color="0.5")
        if savepath:
            fig.savefig(savepath, dpi=200)
        return fig
    sz, su, sv_, serr = sv[:, 0], sv[:, 1], sv[:, 2], sv[:, 3]

    # left: SADCP points over LADCP inverse lines
    ax = axes[0]
    ax.axvline(0, color="0.7", lw=0.8)
    ax.plot(vp.u, vp.z, color=_BLUE, lw=1.7, label="u LADCP")
    ax.plot(vp.v, vp.z, color=_RED, lw=1.7, label="v LADCP")
    ax.errorbar(su, sz, xerr=serr, fmt="o", color=_BLUE, ms=3, lw=0.6, alpha=0.6,
                capsize=0, label="u SADCP")
    ax.errorbar(sv_, sz, xerr=serr, fmt="s", color=_RED, ms=3, lw=0.6, alpha=0.6,
                capsize=0, label="v SADCP")
    ax.invert_yaxis()
    # scale to the velocity signal, not the (occasionally large) error bars
    finite = np.concatenate([vp.u, vp.v, su, sv_])
    finite = finite[np.isfinite(finite)]
    lim = max(0.5, 1.15 * np.nanpercentile(np.abs(finite), 98)) if finite.size else 1.0
    ax.set_xlim(-lim, lim)
    ax.set(xlabel="velocity [m/s]", ylabel="depth [m]")
    ax.legend(fontsize=8, loc="lower left")
    ax.set_title("SADCP (points) vs LADCP inverse (lines)", fontsize=9)

    # right: LADCP - SADCP vs depth over the shared range
    ax = axes[1]
    ax.axvline(0, color="0.7", lw=0.8)
    shared = (sz >= np.nanmin(vp.z)) & (sz <= np.nanmax(vp.z))
    rms = np.nan
    if shared.any():
        lu = np.interp(sz[shared], vp.z, vp.u)
        lv = np.interp(sz[shared], vp.z, vp.v)
        du = lu - su[shared]
        dv = lv - sv_[shared]
        ax.plot(du, sz[shared], "o", color=_BLUE, ms=3, alpha=0.6, label="Δu")
        ax.plot(dv, sz[shared], "s", color=_RED, ms=3, alpha=0.6, label="Δv")
        diff = np.concatenate([du, dv])
        diff = diff[np.isfinite(diff)]
        rms = float(np.sqrt(np.mean(diff ** 2))) if diff.size else np.nan
    ax.invert_yaxis()
    ax.set(xlabel="LADCP − SADCP [m/s]")
    ax.set_xlim(-0.3, 0.3)
    ax.legend(fontsize=8, loc="lower left")
    ax.set_title(f"discrepancy (rms {rms:.3f} m/s)", fontsize=9)

    if own:
        fig.suptitle(f"{station} — ship-ADCP comparison", fontsize=12)
    if savepath:
        fig.savefig(savepath, dpi=200)
    return fig


def sadcp_rms_discrepancy(r: VelocityResult) -> float:
    """Upper-ocean RMS of (LADCP − SADCP) over the shared depth range; NaN if no SADCP."""
    sv = r.sadcp
    if sv is None or len(sv) == 0:
        return np.nan
    sz, su, sv_ = sv[:, 0], sv[:, 1], sv[:, 2]
    shared = (sz >= np.nanmin(r.vp.z)) & (sz <= np.nanmax(r.vp.z))
    if not shared.any():
        return np.nan
    du = np.interp(sz[shared], r.vp.z, r.vp.u) - su[shared]
    dv = np.interp(sz[shared], r.vp.z, r.vp.v) - sv_[shared]
    diff = np.concatenate([du, dv])
    diff = diff[np.isfinite(diff)]
    return float(np.sqrt(np.mean(diff ** 2))) if diff.size else np.nan
