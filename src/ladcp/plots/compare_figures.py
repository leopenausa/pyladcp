"""Figure pages for the ``ladcp-compare`` PDF (pyladcp vs legacy LDEO_IX report).

Both functions draw into an open ``matplotlib`` ``PdfPages`` and consume the
``PairResult`` records built by :mod:`ladcp.qa.compare` (duck-typed here so the
plotting layer carries no import of the pairing logic).
"""
from __future__ import annotations

import numpy as np


def profile_pages(pdf, results, per_page: int = 10):
    """Per-station u/v profile overlays (legacy vs pyladcp ± 1σ, SADCP rows)."""
    import matplotlib.pyplot as plt

    for start in range(0, len(results), per_page):
        chunk = results[start:start + per_page]
        ncol = 5
        nrow = int(np.ceil(len(chunk) / ncol)) * 2
        fig, axes = plt.subplots(nrow, ncol, figsize=(13, 3.4 * nrow),
                                 constrained_layout=True)
        axes = np.atleast_2d(axes)
        for k, r in enumerate(chunk):
            col = k % ncol
            row0 = (k // ncol) * 2
            for off, comp in ((0, "u"), (1, "v")):
                ax = axes[row0 + off, col]
                p = r.profile
                # +-1 sigma solution-uncertainty bands (legacy dr.uerr / our nc uerr;
                # LDEO carries one error profile for both components)
                lfin = np.isfinite(p["lu"]) & np.isfinite(p["luerr"])
                if lfin.any():
                    ax.fill_betweenx(p["lz"][lfin],
                                     p[f"l{comp}"][lfin] - p["luerr"][lfin],
                                     p[f"l{comp}"][lfin] + p["luerr"][lfin],
                                     color="0.3", alpha=0.18, lw=0)
                ofin = np.isfinite(p[comp]) & np.isfinite(p["uerr"])
                if ofin.any():
                    ax.fill_betweenx(p["z"][ofin],
                                     p[comp][ofin] - p["uerr"][ofin],
                                     p[comp][ofin] + p["uerr"][ofin],
                                     color="tab:red", alpha=0.15, lw=0)
                ax.plot(p[f"l{comp}"], p["lz"], "-", color="0.3", lw=1.4,
                        label="legacy ±1σ")
                ax.plot(p[comp], p["z"], "-", color="tab:red", lw=1.0,
                        label="pyladcp ±1σ")
                ci = 1 if comp == "u" else 2
                if p["sadcp"] is not None and p["sadcp"].shape[0]:
                    s = p["sadcp"]
                    ax.plot(s[:, ci], s[:, 0], "o-", color="tab:green", ms=2.5,
                            lw=0.8, alpha=0.85, label="ship ADCP")
                if p["lsz"].size > 1:
                    ax.plot(p["lsu"] if comp == "u" else p["lsv"], p["lsz"], "s",
                            color="tab:olive", ms=2.5, alpha=0.8, mfc="none",
                            label="ship ADCP (legacy constraint)")
                ax.invert_yaxis()
                ax.grid(alpha=0.3)
                s = getattr(r, comp)
                tag = f" [{r.variant}]" if r.variant else ""
                ax.set_title(f"{r.station}{tag} {comp}  r={s.corr:.2f} "
                             f"rms={100 * s.rms:.1f}cm/s", fontsize=8,
                             color="tab:blue" if r.variant else "black")
                if col == 0:
                    ax.set_ylabel("depth [m]")
        for k in range(len(chunk), ncol * (nrow // 2)):
            axes[(k // ncol) * 2, k % ncol].axis("off")
            axes[(k // ncol) * 2 + 1, k % ncol].axis("off")
        from matplotlib.lines import Line2D
        handles = [Line2D([], [], color="0.3", lw=1.4, label="legacy ±1σ"),
                   Line2D([], [], color="tab:red", lw=1.0, label="pyladcp ±1σ")]
        if any(r.profile["sadcp"] is not None and r.profile["sadcp"].shape[0]
               for r in chunk):
            handles.append(Line2D([], [], color="tab:green", marker="o", ms=3,
                                  lw=0.8, label="ship ADCP"))
        if any(r.profile["lsz"].size > 1 for r in chunk):
            handles.append(Line2D([], [], color="tab:olive", marker="s", ms=3,
                                  lw=0, mfc="none",
                                  label="ship ADCP (legacy constraint)"))
        fig.legend(handles=handles, loc="upper left", fontsize=7, ncol=len(handles),
                   frameon=False)
        pdf.savefig(fig)
        plt.close(fig)


def summary_page(pdf, results, title: str):
    """Cruise-level scorecard page: rms/correlation bars, barotropic offsets, coverage."""
    import matplotlib.pyplot as plt

    names = [r.station + ("*" if r.variant else "") for r in results]
    x = np.arange(len(results))
    fig, axes = plt.subplots(2, 2, figsize=(13, 9), constrained_layout=True)

    ax = axes[0, 0]
    ax.bar(x - 0.2, [100 * r.u.rms for r in results], 0.4, label="u")
    ax.bar(x + 0.2, [100 * r.v.rms for r in results], 0.4, label="v")
    ax.set_xticks(x, names, rotation=90, fontsize=7)
    ax.set_ylabel("rms vs legacy [cm/s]")
    ax.legend()
    ax.grid(alpha=0.3, axis="y")

    ax = axes[0, 1]
    ax.bar(x - 0.2, [r.u.corr for r in results], 0.4, label="u")
    ax.bar(x + 0.2, [r.v.corr for r in results], 0.4, label="v")
    ax.set_xticks(x, names, rotation=90, fontsize=7)
    ax.set_ylabel("profile correlation")
    ax.set_ylim(-0.2, 1.0)
    ax.grid(alpha=0.3, axis="y")

    ax = axes[1, 0]
    ax.plot([100 * r.dubar for r in results], [100 * r.dvbar for r in results],
            "o", ms=5)
    ax.axhline(0, color="0.6", lw=0.6)
    ax.axvline(0, color="0.6", lw=0.6)
    ax.set_xlabel("ubar (pyladcp - legacy) [cm/s]")
    ax.set_ylabel("vbar (pyladcp - legacy) [cm/s]")
    ax.set_title("barotropic offset per station", fontsize=9)
    ax.grid(alpha=0.3)

    ax = axes[1, 1]
    ax.plot([r.zmax_legacy for r in results], [r.zmax_ours for r in results],
            "o", ms=5)
    lim = max([*[r.zmax_legacy for r in results],
               *[r.zmax_ours for r in results], 1.0])
    ax.plot([0, lim], [0, lim], "k--", lw=0.8)
    ax.set_xlabel("legacy max depth [m]")
    ax.set_ylabel("pyladcp max depth [m]")
    ax.set_title("depth coverage", fontsize=9)
    ax.grid(alpha=0.3)

    med_u = float(np.nanmedian([r.u.rms for r in results]))
    med_v = float(np.nanmedian([r.v.rms for r in results]))
    med_cu = float(np.nanmedian([r.u.corr for r in results]))
    fig.suptitle(f"{title}\n{len(results)} stations -- median rms "
                 f"u {100 * med_u:.1f} / v {100 * med_v:.1f} cm/s, "
                 f"median corr(u) {med_cu:.2f}", fontsize=12)
    subs = {r.station: r.variant for r in results if r.variant}
    if subs:
        note = "* substituted runs: " + "; ".join(
            f"{st} = {lab}" for st, lab in subs.items())
        fig.text(0.01, 0.005, note, fontsize=8, color="tab:blue")
    pdf.savefig(fig)
    plt.close(fig)
