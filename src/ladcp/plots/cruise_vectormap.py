"""Plan-view velocity vector map from a processed cruise NetCDF — ``ladcp-vectormap``.

The at-a-glance companion to ``ladcp-section``: instead of one vertical slice, draw a map of
**depth-range-integrated velocity vectors**, one arrow per station. Several depth layers can be
stacked in distinct colours on a single map (``--layer 0:200 --layer 200:800``), so the vertical
shear of the horizontal flow is visible at a glance. Each arrow is either the layer's
depth-averaged velocity (``--reduce mean``, m/s) or its depth-integrated transport per unit width
(``--reduce integral``, m²/s = ∫u dz).

Reads the cruise file written by :func:`ladcp.export.netcdf.write_cruise_nc` (via
:func:`ladcp.plots.cruise_section.load_cruise_section`) — no re-solving.

CLI::

    ladcp-vectormap exports/FDCCC1_ladcp.nc --layer 0:200 --layer 200:800 -o map.png

A single occupation is a snapshot whose depth-mean carries the barotropic tide; the figure says
so. Integration is over **depth** ranges only — density-range layers would need a
``density(station, depth)`` variable added to the cruise NetCDF from each station's CTD (future).
"""

from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np

from .cruise_section import CruiseSection, load_cruise_section

# numpy renamed trapz -> trapezoid in 2.0; support both
_trapz = getattr(np, "trapezoid", None) or np.trapz


def _parse_layers(specs: list[str] | None) -> list[tuple[float, float]]:
    """``["0:200", "200:800"]`` → ``[(0,200),(200,800)]``; default full column."""
    if not specs:
        return [(0.0, float("inf"))]
    out = []
    for s in specs:
        lo, _, hi = s.partition(":")
        if not _:
            raise ValueError(f"--layer must be z0:z1 (metres), got {s!r}")
        out.append((float(lo), float(hi or "inf")))
    return out


def layer_reduce(sec: CruiseSection, z0: float, z1: float, reduce: str = "mean"
                 ) -> tuple[np.ndarray, np.ndarray]:
    """Reduce each station's velocity over depth ``[z0, z1]`` to one ``(u, v)`` vector.

    ``mean`` is the depth-average over the finite bins in range [m/s]; ``integral`` is
    ``∫u dz`` (trapezoid over the finite, in-range, above-seabed sub-profile) [m²/s]. Stations
    with fewer than two finite in-range bins (layer below the seabed or unsampled) → NaN.
    """
    z = sec.depth
    ured = np.full(sec.stations.size, np.nan)
    vred = np.full(sec.stations.size, np.nan)
    for i in range(sec.stations.size):
        seabed = sec.bottom_depth[i] if np.isfinite(sec.bottom_depth[i]) else np.inf
        band = (z >= z0) & (z <= z1) & (z <= seabed)
        ok = band & np.isfinite(sec.u[i]) & np.isfinite(sec.v[i])
        if ok.sum() < 2:
            continue
        if reduce == "integral":
            ured[i] = _trapz(sec.u[i][ok], z[ok])
            vred[i] = _trapz(sec.v[i][ok], z[ok])
        else:
            ured[i] = np.nanmean(sec.u[i][ok])
            vred[i] = np.nanmean(sec.v[i][ok])
    return ured, vred


def _layer_label(z0: float, z1: float) -> str:
    hi = "seabed" if not np.isfinite(z1) else f"{z1:g}"
    return f"{z0:g}–{hi} m"


def _fmt_time(t) -> str:
    """``datetime64`` → ``YYYY-MM-DD HH:MM`` UTC, or ``""`` for NaT/missing."""
    if t is None:
        return ""
    t = np.datetime64(t)
    if np.isnat(t):
        return ""
    return np.datetime_as_string(t, unit="m").replace("T", " ")


def cruise_vectormap_figure(sec: CruiseSection, layers: list[tuple[float, float]], *,
                            reduce: str = "mean", stations: list[str] | None = None,
                            scale: float | None = None, coastline: bool = False,
                            show_times: bool = False,
                            title: str = "", fig=None, savepath: str | None = None):
    """Plan-view quiver of depth-range-integrated vectors, one colour per layer."""
    import matplotlib.pyplot as plt

    sel = np.ones(sec.stations.size, bool)
    if stations:
        keep = set(stations)
        sel = np.array([s in keep for s in sec.stations])
    lon, lat = sec.lon[sel], sec.lat[sel]
    units = "m²/s" if reduce == "integral" else "m/s"

    reduced = [(z0, z1, *(a[sel] for a in layer_reduce(sec, z0, z1, reduce)))
               for z0, z1 in layers]
    # one shared scale so arrow lengths are comparable across layers
    mags = np.concatenate([np.hypot(u, v) for _, _, u, v in reduced])
    mags = mags[np.isfinite(mags)]
    ref = max(round(float(np.percentile(mags, 90)), 2), 1e-6) if mags.size else 1.0
    qscale = scale if scale is not None else ref * 12.0

    if fig is None:
        fig = plt.figure(figsize=(10, 8.5), constrained_layout=True)
    ax = fig.add_subplot(111, projection=_carto(coastline))
    if coastline and ax.name == "cartopy":          # pragma: no cover - optional dep
        import cartopy.feature as cfeature
        ax.add_feature(cfeature.GSHHSFeature(scale="full"), facecolor="0.9", edgecolor="0.5")
        tk = {"transform": _platecarree()}
    else:
        tk = {}
        ax.set_aspect(1.0 / np.cos(np.deg2rad(np.nanmedian(lat))))

    ax.scatter(lon, lat, s=14, color="0.5", zorder=3, **tk)
    cmap = plt.get_cmap("tab10")
    for k, (z0, z1, u, v) in enumerate(reduced):
        q = ax.quiver(lon, lat, u, v, color=cmap(k % 10), scale=qscale,
                      width=0.005, zorder=4, label=_layer_label(z0, z1), **tk)
        # keys inside the axes (upper-left) so they survive any map aspect ratio
        ax.quiverkey(q, 0.06, 0.97 - 0.045 * k, ref,
                     f"{_layer_label(z0, z1)}: {ref:g} {units}", labelpos="E",
                     fontproperties={"size": 8}, coordinates="axes")
    times = sec.time[sel] if (show_times and sec.time is not None) else [None] * lon.size
    if not tk:                                       # data-coord labels (non-cartopy path)
        for s, x, y, t in zip(sec.stations[sel], lon, lat, times, strict=True):
            ts = _fmt_time(t)
            lbl = f"{s}\n{ts} UTC" if ts else s
            ax.annotate(lbl, (x, y), fontsize=5, color="0.45", xytext=(2, 2),
                        textcoords="offset points")
    if not tk and np.isfinite(lon).any():            # tight auto-extent (non-cartopy path)
        # centre on the stations; a minimum half-window keeps single-station maps from
        # auto-expanding to a near-empty continental frame, and leaves room for arrows.
        mx, my = 0.5 * (np.nanmin(lon) + np.nanmax(lon)), 0.5 * (np.nanmin(lat) + np.nanmax(lat))
        r = max(np.nanmax(lon) - np.nanmin(lon), np.nanmax(lat) - np.nanmin(lat), 0.12) * 0.65
        ax.set_xlim(mx - r, mx + r); ax.set_ylim(my - r, my + r)
    ax.set_xlabel("longitude [°E]"); ax.set_ylabel("latitude [°N]")
    ax.grid(alpha=0.3)
    ax.set_title(title or "cruise depth-integrated velocity vectors", fontsize=12)
    ax.text(0.5, 0.01, f"arrows = layer {reduce} ({units}); single-occupation snapshot — "
            "depth-mean is tide-aliased", transform=ax.transAxes, ha="center", va="bottom",
            fontsize=8, color="0.4")
    if savepath:
        fig.savefig(savepath, dpi=170)
    return fig


def _carto(coastline: bool):
    """Return a cartopy PlateCarree projection if requested & available, else None."""
    if not coastline:
        return None
    try:                                              # pragma: no cover - optional dep
        return _platecarree()
    except Exception:
        return None


def _platecarree():                                   # pragma: no cover - optional dep
    import cartopy.crs as ccrs
    return ccrs.PlateCarree()


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(
        prog="ladcp-vectormap",
        description="Plan-view depth-integrated velocity vector map from a cruise NetCDF")
    ap.add_argument("cruise", metavar="PATH",
                    help="cruise NetCDF (exports/<CRUISE>_ladcp.nc) or a cruise output dir")
    ap.add_argument("--layer", action="append", metavar="Z0:Z1", default=None,
                    help="depth range [m] to integrate (repeatable; e.g. 0:200 200:800; "
                         "default: full column)")
    ap.add_argument("--reduce", choices=("mean", "integral"), default="mean",
                    help="arrow = depth-averaged velocity [m/s] (default) or depth-integrated "
                         "transport per unit width [m2/s]")
    ap.add_argument("--stations", default=None,
                    help="comma-separated subset of station labels")
    ap.add_argument("--scale", type=float, default=None,
                    help="quiver scale (larger = shorter arrows; default: robust auto)")
    ap.add_argument("--coastline", action="store_true",
                    help="draw a coastline (needs cartopy; ignored if unavailable)")
    ap.add_argument("--show-times", action="store_true",
                    help="annotate each station with its cast start time (UTC) from the NetCDF")
    ap.add_argument("--title", default="", help="figure title")
    ap.add_argument("-o", "--out", default="cruise_vectormap.png", help="output figure path")
    args = ap.parse_args(argv)

    sec = load_cruise_section(args.cruise)
    layers = _parse_layers(args.layer)
    subset = [s.strip() for s in args.stations.split(",")] if args.stations else None
    title = args.title or f"{Path(args.cruise).resolve().stem} velocity vectors"
    cruise_vectormap_figure(sec, layers, reduce=args.reduce, stations=subset,
                            scale=args.scale, coastline=args.coastline,
                            show_times=args.show_times, title=title,
                            savepath=args.out)
    print(f"wrote {args.out}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
