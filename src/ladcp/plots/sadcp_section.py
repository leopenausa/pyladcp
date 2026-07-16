"""Shipboard-ADCP velocity sections — depth vs time or along-track distance.

Renders the absolute ocean velocity carried by a :class:`~ladcp.io.sadcp_vmdas.
SadcpDataset` (raw VmDAS via ``--source vmdas``, CODAS-processed NetCDF via
``--source codas``) as u/v pcolormesh panels, with optional LADCP station ticks
from the cruise archive index.

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

from ..io.sadcp_types import SadcpDataset
from .section_grid import along_track_km, auto_clim, station_ticks, velocity_panel

_ANOM_MIN_BINS = 5


def depth_mean_anomaly(comp: np.ndarray, zmask: np.ndarray) -> np.ndarray:
    """``comp`` minus its per-ensemble depth mean over the ``zmask`` bins.

    Columns with fewer than ``_ANOM_MIN_BINS`` finite bins in the window go
    all-NaN (a mean of 1-2 cells is not a barotropic estimate).
    """
    import warnings
    sub = comp[zmask]
    nfin = np.sum(np.isfinite(sub), axis=0)
    with warnings.catch_warnings():
        warnings.simplefilter("ignore", RuntimeWarning)  # all-NaN columns
        cmean = np.nanmean(sub, axis=0)
    cmean[nfin < _ANOM_MIN_BINS] = np.nan
    return comp - cmean[None, :]


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
                         anomaly: bool = False,
                         stations=None, title: str = "",
                         fig=None, savepath: str | None = None):
    """u/v section panels vs time or along-track distance.

    ``stations`` is an optional list of ``(label, utc)`` tuples (e.g. LADCP casts
    from the archive index) drawn as ticks along the top of each panel. ``clim``
    fixes the symmetric colour range [m/s]; default is the robust 98th percentile.

    ``anomaly`` subtracts each ensemble's depth-mean velocity (over the plotted
    depth range) before rendering — a baroclinic "shear section". Use it when a
    barotropic flow visually swamps the vertical structure: at MORIA the
    within-column std (~7-8 cm/s) matches the depth-mean signal, invisible at a
    ±0.3 m/s clim.

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

    if anomaly:
        sub.u = depth_mean_anomaly(sub.u, zmask)
        sub.v = depth_mean_anomaly(sub.v, zmask)

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
        clim = auto_clim([sub.u[zmask], sub.v[zmask]])

    if fig is None:
        fig = plt.figure(figsize=(11, 7), constrained_layout=True)
    axes = fig.subplots(2, 1, sharex=True)

    names = (("u′ (east, minus depth mean)", "v′ (north, minus depth mean)")
             if anomaly else ("u (east)", "v (north)"))
    for ax, comp, name in ((axes[0], sub.u, names[0]), (axes[1], sub.v, names[1])):
        velocity_panel(fig, ax, x, z, comp[zmask], clim=clim,
                       title=f"{name}  [{sub.freq_khz} kHz {sub.file_type}]")
        station_ticks(ax, marks, annotate=(ax is axes[0]),
                      color="0.25", lw=0.5, alpha=0.7, dy=10)
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
    d = json.load(open(index_path, encoding="utf-8"))
    casts = d.get("casts", d)
    return sorted(((st, c["utc"]) for st, c in casts.items() if c.get("utc")),
                  key=lambda s: s[1])


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(
        prog="ladcp-sadcp-section",
        description="Shipboard-ADCP u/v sections (depth vs time or distance)")
    ap.add_argument("--sadcp", required=True, metavar="PATH",
                    help="VmDAS folder (STA/LTA; ingested once and cached) or, with "
                         "--source codas, a CODAS contour NetCDF (file or processing dir)")
    ap.add_argument("--source", choices=("vmdas", "codas"), default="vmdas",
                    help="what --sadcp points at: raw VmDAS averages (default) or a "
                         "CODAS-processed NetCDF product")
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
    ap.add_argument("--anomaly", action="store_true",
                    help="subtract each ensemble's depth-mean velocity (baroclinic "
                         "shear section; reveals vertical structure under a "
                         "barotropic flow)")
    ap.add_argument("--index", default=None,
                    help="archive index JSON: draw LADCP station ticks")
    ap.add_argument("--title", default="", help="figure title")
    ap.add_argument("-o", "--out", default="sadcp_section.png",
                    help="output PNG (default sadcp_section.png)")
    args = ap.parse_args(argv)

    if args.source == "codas":
        from ..io.sadcp_codas import read_codas_nc
        ds = read_codas_nc(args.sadcp)
    else:
        from ..io.sadcp_vmdas import load_or_ingest
        ds = load_or_ingest(args.sadcp, file_type=args.filetype,
                            transducer_depth=args.xducer)
    stations = _index_stations(args.index) if args.index else None
    title = args.title or f"{Path(args.sadcp).resolve().parent.name} ship-ADCP section"
    sadcp_section_figure(ds, by=args.by, time_start=args.start, time_end=args.end,
                         max_depth=args.max_depth, clim=args.clim,
                         speed_max=args.speed_max, anomaly=args.anomaly,
                         stations=stations, title=title, savepath=args.out)
    print(f"wrote {args.out}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
