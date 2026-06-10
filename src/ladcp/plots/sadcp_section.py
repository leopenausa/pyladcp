"""Shipboard-ADCP velocity sections — depth vs time or along-track distance.

Renders the absolute ocean velocity carried by a :class:`~ladcp.io.sadcp_vmdas.
SadcpDataset` (raw VmDAS today; the CODAS-processed reader will produce the same
dataset, so these sections work for both sources) as u/v pcolormesh panels, with
optional LADCP station ticks from the cruise archive index.

CLI::

    ladcp-sadcp-section --sadcp "$B/sADCP/sadcp_150/DATA" \
        --index "$B/.ladcp_archive.json" --by distance -o sections/

``--by time`` plots against the ship clock (natural for station work);
``--by distance`` against cumulative along-track distance (natural for transects).
"""

from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np

from ..io.sadcp_vmdas import SadcpDataset

_M_PER_DEG_LAT = 110_540.0
_M_PER_DEG_LON = 111_320.0


def along_track_km(lat: np.ndarray, lon: np.ndarray) -> np.ndarray:
    """Cumulative along-track distance [km] from per-ensemble nav (NaN-tolerant).

    Equirectangular increments (exact enough at ship scales); NaN fixes contribute
    zero step so the axis stays monotonic non-decreasing.
    """
    la = np.asarray(lat, float)
    lo = np.asarray(lon, float)
    latm = np.nanmedian(la)
    dx = np.diff(lo) * _M_PER_DEG_LON * np.cos(np.deg2rad(latm))
    dy = np.diff(la) * _M_PER_DEG_LAT
    step = np.hypot(dx, dy)
    step[~np.isfinite(step)] = 0.0
    return np.concatenate([[0.0], np.cumsum(step)]) / 1000.0


def _station_marks(stations, ds: SadcpDataset, x: np.ndarray):
    """Map (label, utc) station entries onto the section x-axis.

    Returns ``[(x_pos, label), ...]`` for stations whose time falls inside the
    plotted window (nearest-ensemble mapping, so it works for both axis kinds).
    """
    out = []
    t = ds.time.astype("datetime64[ns]").astype("int64")
    for label, utc in stations:
        ti = np.datetime64(utc).astype("datetime64[ns]").astype("int64")
        if ti < t[0] or ti > t[-1]:
            continue
        i = int(np.argmin(np.abs(t - ti)))
        out.append((x[i], label))
    return out


def sadcp_section_figure(ds: SadcpDataset, *, by: str = "time",
                         time_start=None, time_end=None,
                         max_depth: float | None = None,
                         clim: float | None = None,
                         speed_max: float = 1.5,
                         stations=None, title: str = "",
                         fig=None, savepath: str | None = None):
    """u/v section panels vs time or along-track distance.

    ``stations`` is an optional list of ``(label, utc)`` tuples (e.g. LADCP casts
    from the archive index) drawn as ticks along the top of each panel. ``clim``
    fixes the symmetric colour range [m/s]; default is the robust 98th percentile.

    ``speed_max`` blanks whole ensembles whose depth-median speed exceeds it
    (default 1.5 m/s): raw VmDAS carries occasional stuck/poor GPS fixes whose
    uncorrected ship velocity leaks into the "ocean" velocity at full ship speed
    (~4% of MORIA ensembles). Set 0 to disable (e.g. for CODAS-cleaned data).
    """
    import matplotlib.dates as mdates
    import matplotlib.pyplot as plt

    if by not in ("time", "distance"):
        raise ValueError(f"by must be 'time' or 'distance', got {by!r}")

    sel = np.ones(ds.n_ens, dtype=bool)
    if time_start is not None:
        sel &= ds.time >= np.datetime64(time_start)
    if time_end is not None:
        sel &= ds.time <= np.datetime64(time_end)
    if sel.sum() < 2:
        raise ValueError("fewer than 2 ensembles in the requested time window")
    sub = SadcpDataset(time=ds.time[sel], lat=ds.lat[sel], lon=ds.lon[sel],
                       depth=ds.depth, u=ds.u[:, sel].copy(), v=ds.v[:, sel].copy(),
                       freq_khz=ds.freq_khz, transducer_depth=ds.transducer_depth,
                       file_type=ds.file_type, source=ds.source, n_files=ds.n_files)

    n_screened = 0
    if speed_max and speed_max > 0:
        with np.errstate(invalid="ignore"):
            spd = np.nanmedian(np.hypot(sub.u, sub.v), axis=0)
        leak = spd > speed_max
        n_screened = int(leak.sum())
        sub.u[:, leak] = np.nan
        sub.v[:, leak] = np.nan

    zmask = np.ones(sub.depth.size, dtype=bool)
    if max_depth is not None:
        zmask = sub.depth <= max_depth
    z = sub.depth[zmask]

    if by == "distance":
        x = along_track_km(sub.lat, sub.lon)
        xlabel = "along-track distance [km]"
    else:
        x = mdates.date2num(sub.time.astype("datetime64[ms]").astype("O"))
        xlabel = "time (UTC)"

    # station marks map cast times onto ensemble indices -> must use the pre-insertion x
    marks = _station_marks(stations, sub, x) if stations else []

    # break recording gaps: 'nearest' shading would stretch isolated ensembles across
    # hours-long off periods. Insert an all-NaN column at the centre of any x-gap wider
    # than 10x the median spacing so pcolormesh leaves the gap blank.
    dx = np.diff(x)
    med_dx = np.median(dx[dx > 0]) if np.any(dx > 0) else 0.0
    gaps = np.where(dx > 10 * max(med_dx, 1e-9))[0]
    if gaps.size:
        x = np.insert(x, gaps + 1, x[gaps] + dx[gaps] / 2)
        nan_col = np.full((sub.depth.size, 1), np.nan)
        for k, g in enumerate(gaps):
            sub.u = np.insert(sub.u, g + 1 + k, nan_col[:, 0], axis=1)
            sub.v = np.insert(sub.v, g + 1 + k, nan_col[:, 0], axis=1)

    if clim is None:
        finite = np.abs(np.concatenate([sub.u[zmask].ravel(), sub.v[zmask].ravel()]))
        finite = finite[np.isfinite(finite)]
        clim = float(np.percentile(finite, 98)) if finite.size else 0.5
        clim = max(round(clim, 2), 0.05)

    if fig is None:
        fig = plt.figure(figsize=(11, 7), constrained_layout=True)
    axes = fig.subplots(2, 1, sharex=True)

    for ax, comp, name in ((axes[0], sub.u, "u (east)"), (axes[1], sub.v, "v (north)")):
        pm = ax.pcolormesh(x, z, comp[zmask], cmap="RdBu_r", vmin=-clim, vmax=clim,
                           shading="nearest")
        ax.invert_yaxis()
        ax.set_ylabel("depth [m]")
        ax.set_title(f"{name}  [{sub.freq_khz} kHz {sub.file_type}]", fontsize=9)
        fig.colorbar(pm, ax=ax, label="m/s", pad=0.01)
        for xm, _label in marks:
            ax.axvline(xm, color="0.25", lw=0.5, ls=":", alpha=0.7)
        if marks and ax is axes[0]:
            for xm, label in marks:
                ax.annotate(label, (xm, 0), xytext=(0, 10), textcoords="offset points",
                            ha="center", fontsize=6, rotation=90, color="0.25",
                            annotation_clip=False)
    axes[1].set_xlabel(xlabel)
    if by == "time":
        axes[1].xaxis.set_major_formatter(mdates.DateFormatter("%m-%d %H:%M"))
        fig.autofmt_xdate(rotation=30)
    if n_screened:
        axes[1].annotate(f"{n_screened} ensembles blanked (median speed > "
                         f"{speed_max:g} m/s: ship-velocity leak)", (0.99, -0.32),
                         xycoords="axes fraction", ha="right", fontsize=7, color="0.4")
    if title:
        fig.suptitle(title, fontsize=12)
    if savepath:
        fig.savefig(savepath, dpi=170)
    return fig


def _index_stations(index_path: str):
    """LADCP casts from an archive index as ``(label, utc)`` (skips undated)."""
    import json
    d = json.load(open(index_path))
    casts = d.get("casts", d)
    return sorted(((st, c["utc"]) for st, c in casts.items() if c.get("utc")),
                  key=lambda s: s[1])


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(
        prog="ladcp-sadcp-section",
        description="Shipboard-ADCP u/v sections (depth vs time or distance)")
    ap.add_argument("--sadcp", required=True, metavar="DIR",
                    help="VmDAS folder (STA/LTA; ingested once and cached)")
    ap.add_argument("--by", choices=("time", "distance"), default="time",
                    help="x-axis: ship clock (default) or along-track distance")
    ap.add_argument("--filetype", choices=("STA", "LTA"), default="STA")
    ap.add_argument("--xducer", type=float, default=5.0,
                    help="transducer depth below waterline [m] (default 5)")
    ap.add_argument("--start", default=None, help="window start (ISO, e.g. 2025-10-03)")
    ap.add_argument("--end", default=None, help="window end (ISO)")
    ap.add_argument("--max-depth", type=float, default=None, help="crop depth axis [m]")
    ap.add_argument("--clim", type=float, default=None,
                    help="symmetric colour range [m/s] (default: robust auto)")
    ap.add_argument("--speed-max", type=float, default=1.5,
                    help="blank ensembles whose median speed exceeds this [m/s] "
                         "(GPS-leak screen; 0 disables; default 1.5)")
    ap.add_argument("--index", default=None,
                    help="archive index JSON: draw LADCP station ticks")
    ap.add_argument("--title", default="", help="figure title")
    ap.add_argument("-o", "--out", default="sadcp_section.png",
                    help="output PNG (default sadcp_section.png)")
    args = ap.parse_args(argv)

    from ..io.sadcp_vmdas import load_or_ingest
    ds = load_or_ingest(args.sadcp, file_type=args.filetype,
                        transducer_depth=args.xducer)
    stations = _index_stations(args.index) if args.index else None
    title = args.title or f"{Path(args.sadcp).resolve().parent.name} ship-ADCP section"
    sadcp_section_figure(ds, by=args.by, time_start=args.start, time_end=args.end,
                         max_depth=args.max_depth, clim=args.clim,
                         speed_max=args.speed_max,
                         stations=stations, title=title, savepath=args.out)
    print(f"wrote {args.out}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
