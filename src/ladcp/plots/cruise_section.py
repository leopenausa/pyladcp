"""Cruise velocity sections from a processed cruise NetCDF — ``ladcp-section``.

Reads the self-contained cruise file written by :func:`ladcp.export.netcdf.write_cruise_nc`
(``(station, depth)`` grids of ``u``/``v``/``uerr`` plus per-station ``lat``/``lon``/``time``
/``bottom_depth``) and renders a depth-vs-distance velocity section: objective-analysis-gridded
contours that are smooth where the survey sampled and blank where it did not, the real seabed
drawn underneath, and stations ticked on top.

Velocity is shown both as earth ``u``/``v`` and rotated into **cross-/along-track** components
(the transport-relevant pair) using a per-segment transect azimuth, so curved/dog-leg surveys
rotate faithfully.

CLI::

    ladcp-section exports/FDCCC1_ladcp.nc --component both -o section.pdf

A single occupation is a snapshot whose depth-mean is tide-aliased; the figure says so. The
plan-view vector-map companion (depth-range-integrated arrows) is the ``ladcp-vectormap``
command in :mod:`ladcp.plots.cruise_vectormap`.
"""

from __future__ import annotations

import argparse
from dataclasses import dataclass
from pathlib import Path

import numpy as np

from .sadcp_section import along_track_km
from .section_grid import auto_oa_params, grid_linear, grid_oa

_M_PER_DEG_LAT = 110_540.0
_M_PER_DEG_LON = 111_320.0


@dataclass
class CruiseSection:
    """Per-station velocity grids read back from a cruise NetCDF."""

    stations: np.ndarray        # [ns] station labels
    depth: np.ndarray           # [nz] depth grid [m, +down]
    u: np.ndarray               # [ns, nz] eastward velocity [m/s]
    v: np.ndarray               # [ns, nz] northward velocity [m/s]
    uerr: np.ndarray            # [ns, nz] velocity uncertainty [m/s]
    lat: np.ndarray             # [ns]
    lon: np.ndarray             # [ns]
    bottom_depth: np.ndarray    # [ns] seabed depth [m]


def load_cruise_section(path: str) -> CruiseSection:
    """Load a cruise NetCDF (or a cruise dir holding ``exports/*_ladcp.nc``)."""
    import xarray as xr

    p = Path(path)
    if p.is_dir():
        hits = sorted(p.glob("exports/*_ladcp.nc")) or sorted(p.glob("*_ladcp.nc"))
        if not hits:
            raise FileNotFoundError(f"no cruise NetCDF (exports/*_ladcp.nc) under {p}")
        p = hits[0]
    with xr.open_dataset(p) as ds:
        return CruiseSection(
            stations=np.asarray(ds["station"].values).astype(str),
            depth=np.asarray(ds["depth"].values, float),
            u=np.asarray(ds["u"].values, float),
            v=np.asarray(ds["v"].values, float),
            uerr=np.asarray(ds["uerr"].values, float) if "uerr" in ds else
            np.full(ds["u"].shape, np.nan),
            lat=np.asarray(ds["latitude"].values, float),
            lon=np.asarray(ds["longitude"].values, float),
            bottom_depth=np.asarray(ds["bottom_depth"].values, float) if "bottom_depth" in ds
            else np.full(ds.sizes["station"], np.nan),
        )


def _transect_order(lat: np.ndarray, lon: np.ndarray) -> np.ndarray:
    """Order stations along the transect: projection onto their principal axis.

    Works for any transect orientation (the first principal component of the station
    positions is the section line); ties/curvature are handled gracefully.
    """
    latm = np.nanmedian(lat)
    x = (lon - np.nanmean(lon)) * _M_PER_DEG_LON * np.cos(np.deg2rad(latm))
    y = (lat - np.nanmean(lat)) * _M_PER_DEG_LAT
    pts = np.column_stack([x, y])
    good = np.isfinite(pts).all(axis=1)
    if good.sum() < 2:
        return np.arange(lat.size)
    cov = np.cov(pts[good].T)
    _, vecs = np.linalg.eigh(cov)
    axis = vecs[:, -1]                       # largest-eigenvalue direction
    proj = pts @ axis
    proj[~good] = np.inf                     # park undefined positions at the end
    return np.argsort(proj)


def _segment_bearings(lat: np.ndarray, lon: np.ndarray) -> np.ndarray:
    """Per-station transect heading [deg, from north, clockwise], averaged over adjacent
    segments so a dog-leg rotates faithfully at the bend."""
    la = np.deg2rad(lat)
    lo = np.deg2rad(lon)
    dlon = np.diff(lo)
    x = np.sin(dlon) * np.cos(la[1:])
    y = np.cos(la[:-1]) * np.sin(la[1:]) - np.sin(la[:-1]) * np.cos(la[1:]) * np.cos(dlon)
    seg = np.arctan2(x, y)                    # [n-1] segment bearings (rad)
    # per-station heading = circular mean of adjacent segment bearings
    head = np.empty(lat.size)
    sin_b, cos_b = np.sin(seg), np.cos(seg)
    head[0] = seg[0]
    head[-1] = seg[-1]
    if lat.size > 2:
        head[1:-1] = np.arctan2(sin_b[:-1] + sin_b[1:], cos_b[:-1] + cos_b[1:])
    return np.rad2deg(head) % 360.0


def rotate_cross_along(u: np.ndarray, v: np.ndarray, heading_deg: np.ndarray
                       ) -> tuple[np.ndarray, np.ndarray]:
    """Rotate earth ``(u, v)`` into ``(cross, along)`` for a per-station ``heading``.

    ``along`` is positive in the direction of travel; ``cross`` is positive to the **left**
    of the heading. For a due-east transect (heading 90°): along = u (east), cross = v (north).
    ``heading_deg`` broadcasts over the depth axis (shape ``[ns, 1]``).
    """
    th = np.deg2rad(np.asarray(heading_deg, float))[:, None]
    along = u * np.sin(th) + v * np.cos(th)
    cross = -u * np.cos(th) + v * np.sin(th)
    return cross, along


def _depth_mean_anomaly(comp: np.ndarray) -> np.ndarray:
    """Per-station (row) velocity minus its own depth mean (baroclinic anomaly)."""
    import warnings
    with warnings.catch_warnings():
        warnings.simplefilter("ignore", RuntimeWarning)
        m = np.nanmean(comp, axis=1)
    return comp - m[:, None]


def cruise_section_figure(sec: CruiseSection, *, component: str = "both",
                          gridding: str = "oa", anomaly: bool = False,
                          order: str = "distance", stations: list[str] | None = None,
                          azimuth: float | None = None, clim: float | None = None,
                          lh: float | None = None, lv: float | None = None,
                          max_depth: float | None = None, title: str = "",
                          fig=None, savepath: str | None = None):
    """Render the cruise velocity section. Returns the matplotlib Figure."""
    import matplotlib.pyplot as plt

    # --- station selection + ordering ---
    idx = np.arange(sec.stations.size)
    if stations:
        keep = {s for s in stations}
        idx = np.array([i for i in idx if sec.stations[i] in keep])
        if idx.size < 2:
            raise ValueError("need at least 2 of the requested stations in the cruise file")
    lat, lon = sec.lat[idx], sec.lon[idx]
    if order == "distance":
        o = _transect_order(lat, lon)
    elif order == "station":
        o = np.argsort(sec.stations[idx])
    else:                                    # time order = file order (already by time)
        o = np.arange(idx.size)
    idx = idx[o]
    labels = sec.stations[idx]
    lat, lon = sec.lat[idx], sec.lon[idx]
    u, v = sec.u[idx], sec.v[idx]
    bottom = sec.bottom_depth[idx]

    x_st = along_track_km(lat, lon)
    heading = np.full(idx.size, azimuth) if azimuth is not None else _segment_bearings(lat, lon)

    # --- components to draw ---
    cross, along = rotate_cross_along(u, v, heading)
    panels: list[tuple[np.ndarray, str]] = []
    if component in ("cross-along", "both"):
        panels += [(cross, "cross-track (+left of track)"), (along, "along-track (+downstream)")]
    if component in ("uv", "both"):
        panels += [(u, "u (east)"), (v, "v (north)")]
    if anomaly:
        panels = [(_depth_mean_anomaly(c), f"{n} − depth mean") for c, n in panels]

    # --- grid geometry: dense distance axis + seabed line from real bottom depths ---
    z = sec.depth
    if max_depth is not None:
        z = z[z <= max_depth]
    x_dense = np.linspace(float(x_st[0]), float(x_st[-1]), 300)
    bd_finite = np.isfinite(bottom)
    bathy_dense = (np.interp(x_dense, x_st[bd_finite], bottom[bd_finite])
                   if bd_finite.sum() >= 2 else np.full_like(x_dense, np.nanmax(z)))
    if lh is None or lv is None:
        alh, alv = auto_oa_params(x_st, float(np.nanmax(z)))
        lh = lh or alh
        lv = lv or alv

    def _grid(comp: np.ndarray) -> np.ndarray:
        profiles = [{"dep": z, "val": comp[j, : z.size]} for j in range(idx.size)]
        if gridding == "linear":
            return grid_linear(profiles, x_dense, z, x_st, bathy_dense)
        return grid_oa(profiles, x_dense, z, x_st, bathy_dense, lh, lv)

    grids = [(_grid(c), n) for c, n in panels]

    if clim is None:
        allv = np.abs(np.concatenate([g.ravel() for g, _ in grids]))
        allv = allv[np.isfinite(allv)]
        clim = max(round(float(np.percentile(allv, 98)), 2), 0.05) if allv.size else 0.5

    n = len(grids)
    if fig is None:
        fig = plt.figure(figsize=(11, 2.6 * n + 0.8), constrained_layout=True)
    axes = np.atleast_1d(fig.subplots(n, 1, sharex=True))
    for ax, (g, name) in zip(axes, grids, strict=True):
        pm = ax.pcolormesh(x_dense, z, g, cmap="RdBu_r", vmin=-clim, vmax=clim,
                           shading="nearest")
        ax.plot(x_dense, bathy_dense, color="0.2", lw=1.2)           # seabed line
        ax.fill_between(x_dense, bathy_dense, z.max(), color="0.75", zorder=0)
        ax.invert_yaxis()
        ax.set_ylabel("depth [m]")
        ax.set_title(name, fontsize=9)
        fig.colorbar(pm, ax=ax, label="m/s", pad=0.01)
        for xm in x_st:
            ax.axvline(xm, color="0.3", lw=0.4, ls=":", alpha=0.6)
        if ax is axes[0]:
            for xm, lab in zip(x_st, labels, strict=True):
                ax.annotate(lab, (xm, 0), xytext=(0, 8), textcoords="offset points",
                            ha="center", fontsize=6, rotation=90, color="0.3",
                            annotation_clip=False)
    axes[-1].set_xlabel("along-track distance [km]")
    axes[-1].annotate("single-occupation snapshot — depth-mean is tide-aliased; "
                      "contours objective-analysis gridded (blank where unsampled)",
                      (0.99, -0.34), xycoords="axes fraction", ha="right", fontsize=7,
                      color="0.4")
    if title:
        fig.suptitle(title, fontsize=12)
    if savepath:
        fig.savefig(savepath, dpi=170)
    return fig


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(
        prog="ladcp-section",
        description="Cruise velocity section from a processed cruise NetCDF")
    ap.add_argument("cruise", metavar="PATH",
                    help="cruise NetCDF (exports/<CRUISE>_ladcp.nc) or a cruise output dir")
    ap.add_argument("--component", choices=("both", "cross-along", "uv"), default="both",
                    help="velocity components to draw (default: both)")
    ap.add_argument("--gridding", choices=("oa", "linear"), default="oa",
                    help="objective-analysis (default) or linear interpolation")
    ap.add_argument("--anomaly", action="store_true",
                    help="subtract each station's depth-mean (baroclinic section)")
    ap.add_argument("--order", choices=("distance", "station", "time"), default="distance",
                    help="station ordering along the x-axis (default: distance)")
    ap.add_argument("--stations", default=None,
                    help="comma-separated subset of station labels (one transect/leg)")
    ap.add_argument("--azimuth", type=float, default=None,
                    help="fixed transect heading [deg from N] for cross/along rotation "
                         "(default: per-segment from station positions)")
    ap.add_argument("--clim", type=float, default=None,
                    help="symmetric colour range [m/s] (default: robust auto)")
    ap.add_argument("--lh", type=float, default=None, help="OA horizontal length [km]")
    ap.add_argument("--lv", type=float, default=None, help="OA vertical length [m]")
    ap.add_argument("--max-depth", type=float, default=None, help="crop depth axis [m]")
    ap.add_argument("--title", default="", help="figure title")
    ap.add_argument("-o", "--out", default="cruise_section.png", help="output figure path")
    args = ap.parse_args(argv)

    sec = load_cruise_section(args.cruise)
    subset = [s.strip() for s in args.stations.split(",")] if args.stations else None
    title = args.title or f"{Path(args.cruise).resolve().stem} velocity section"
    cruise_section_figure(sec, component=args.component, gridding=args.gridding,
                          anomaly=args.anomaly, order=args.order, stations=subset,
                          azimuth=args.azimuth, clim=args.clim, lh=args.lh, lv=args.lv,
                          max_depth=args.max_depth, title=title, savepath=args.out)
    print(f"wrote {args.out}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
