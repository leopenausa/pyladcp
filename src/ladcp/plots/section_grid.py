"""Shared section-plot machinery: OA gridding, distance axis, panels, colour range.

Used by both section renderers (``cruise_section`` — LADCP profiles from the cruise
NetCDF — and ``sadcp_section`` — ship-ADCP ensembles), so the along-track axis, the
robust colour range, and the velocity-panel look cannot drift between them.

The gridding half is bathymetry-aware objective analysis for cruise sections.

A self-contained port of the anisotropic-Gaussian objective-analysis gridder used in the
CTD pipeline's section plots (vendored here so pyladcp carries no cross-project dependency
and needs no network bathymetry — the seafloor comes from the cruise's own per-station
``bottom_depth``). Despite the upstream "DIVA" name this is *objective analysis*, not the
finite-element variational scheme: a separable Gaussian-weighted average at every grid node,

    f̂(x,z) = Σ wᵢ·vᵢ / Σ wᵢ ,   wᵢ = exp(−((x−xᵢ)/Lh)² − ((z−zᵢ)/Lv)²)

collapsed to two matrix multiplies. The honesty guards are the point: a 3·Lh horizontal
cutoff blanks nodes far from any cast, the surface mask refuses to extrapolate above the
shallowest nearby observation, and the bathymetry mask blanks below the seafloor — so the
field is smooth where the survey sampled and NaN where it did not.
"""

from __future__ import annotations

import numpy as np

# coarse per-degree scales for plot axes and station ordering (NOT the velocity-critical
# constant in io/sadcp_vmdas.py, which is golden-pinned)
M_PER_DEG_LAT = 110_540.0
M_PER_DEG_LON = 111_320.0


def along_track_km(lat: np.ndarray, lon: np.ndarray) -> np.ndarray:
    """Cumulative along-track distance [km] from per-ensemble nav (NaN-tolerant).

    Equirectangular increments (exact enough at ship scales); NaN fixes contribute
    zero step so the axis stays monotonic non-decreasing.
    """
    la = np.asarray(lat, float)
    lo = np.asarray(lon, float)
    latm = np.nanmedian(la)
    dx = np.diff(lo) * M_PER_DEG_LON * np.cos(np.deg2rad(latm))
    dy = np.diff(la) * M_PER_DEG_LAT
    step = np.hypot(dx, dy)
    step[~np.isfinite(step)] = 0.0
    return np.concatenate([[0.0], np.cumsum(step)]) / 1000.0


def auto_clim(values, fallback: float = 0.5) -> float:
    """Robust symmetric colour range [m/s]: 98th percentile of |values|, floored at 0.05."""
    a = np.abs(np.concatenate([np.ravel(v) for v in values]))
    a = a[np.isfinite(a)]
    return max(round(float(np.percentile(a, 98)), 2), 0.05) if a.size else fallback


def velocity_panel(fig, ax, x, z, comp, *, clim: float, title: str):
    """One RdBu velocity pcolormesh panel with the shared section look."""
    pm = ax.pcolormesh(x, z, comp, cmap="RdBu_r", vmin=-clim, vmax=clim,
                       shading="nearest")
    ax.invert_yaxis()
    ax.set_ylabel("depth [m]")
    ax.set_title(title, fontsize=9)
    fig.colorbar(pm, ax=ax, label="m/s", pad=0.01)
    return pm


def station_ticks(ax, marks, *, annotate: bool = False, color: str = "0.3",
                  lw: float = 0.4, alpha: float = 0.6, dy: int = 8) -> None:
    """Dotted station verticals from ``[(x, label), ...]``; labels above when ``annotate``."""
    for xm, _label in marks:
        ax.axvline(xm, color=color, lw=lw, ls=":", alpha=alpha)
    if annotate:
        for xm, label in marks:
            if label is None:
                continue
            ax.annotate(label, (xm, 0), xytext=(0, dy), textcoords="offset points",
                        ha="center", fontsize=6, rotation=90, color=color,
                        annotation_clip=False)


def auto_oa_params(x_st: np.ndarray, max_depth: float) -> tuple[float, float]:
    """Default correlation lengths from transect geometry: ``(Lh, Lv)``.

    ``Lh`` (x-axis units, i.e. km) is the median inter-station spacing, floored at 0.5;
    ``Lv`` (metres) is ``max_depth/10`` clamped to ``[10, 50]``.
    """
    x_st = np.asarray(x_st, float)
    lh = float(np.median(np.diff(np.sort(x_st)))) if x_st.size >= 2 else 50.0
    lh = max(lh, 0.5)
    lv = float(np.clip(max_depth / 10.0, 10.0, 50.0))
    return lh, lv


def grid_oa(profiles: list[dict], x_grid: np.ndarray, z_grid: np.ndarray,
            x_st: np.ndarray, bathy_dense: np.ndarray,
            lh: float, lv: float) -> np.ndarray:
    """Objective-analysis grid of scattered casts onto ``(z_grid, x_grid)``.

    ``profiles`` is one dict per station with finite-masked ``dep`` (m, +down) and ``val``
    arrays; ``x_st`` their x-positions (km). ``bathy_dense`` is the seafloor depth at each
    ``x_grid`` node (m). Returns ``Z`` shaped ``(n_z, n_x)`` with NaN outside the sampled
    envelope / below the seabed.
    """
    obs_xs, obs_zs, obs_vs = [], [], []
    for p, xi in zip(profiles, x_st, strict=True):
        ok = np.isfinite(p["val"]) & np.isfinite(p["dep"])
        if not ok.any():
            continue
        n = int(ok.sum())
        obs_xs.append(np.full(n, xi, dtype=np.float64))
        obs_zs.append(np.asarray(p["dep"], float)[ok])
        obs_vs.append(np.asarray(p["val"], float)[ok])

    if not obs_xs:
        return np.full((len(z_grid), len(x_grid)), np.nan)

    obs_x = np.concatenate(obs_xs)
    obs_z = np.concatenate(obs_zs)
    obs_v = np.concatenate(obs_vs)

    cutoff_h = 3.0 * lh
    dx = obs_x[:, None] - np.asarray(x_grid, float)[None, :]      # (n_obs, n_x)
    within_h = np.abs(dx) <= cutoff_h
    dx2 = dx ** 2 / (lh ** 2)
    dx2[~within_h] = np.inf                                        # exp(-inf) = 0
    wx = np.exp(-dx2)
    dz2 = (obs_z[:, None] - np.asarray(z_grid, float)[None, :]) ** 2 / (lv ** 2)
    wz = np.exp(-dz2)                                              # (n_obs, n_z)

    den = wz.T @ wx                                               # (n_z, n_x)
    num = (obs_v[:, None] * wz).T @ wx
    with np.errstate(invalid="ignore", divide="ignore"):
        z = np.where(den > 1e-10, num / den, np.nan)

    # surface mask: no extrapolation above the shallowest observation within the cutoff
    obs_z_in = np.where(within_h, obs_z[:, None], np.inf)
    z_top_col = obs_z_in.min(axis=0)
    z[z_grid[:, None] < z_top_col[None, :]] = np.nan
    # bathymetry mask: blank below the seafloor and on land/undefined columns
    bd = np.asarray(bathy_dense, float)
    with np.errstate(invalid="ignore"):
        z[z_grid[:, None] > bd[None, :]] = np.nan
        z[:, ~(bd > 0.0)] = np.nan
    return z


def grid_linear(profiles: list[dict], x_grid: np.ndarray, z_grid: np.ndarray,
                x_st: np.ndarray, bathy_dense: np.ndarray) -> np.ndarray:
    """Linear-interpolation fallback: each depth row interpolated across stations.

    Honest at the ends (no horizontal extrapolation beyond the outer casts) and blanked
    below the seafloor, matching :func:`grid_oa`'s masking but without smoothing.
    """
    x_st = np.asarray(x_st, float)
    order = np.argsort(x_st)
    xs = x_st[order]
    # station-major value matrix on the common z_grid
    vals = np.full((len(z_grid), len(x_st)), np.nan)
    for j, p in enumerate(profiles):
        vals[:, j] = np.interp(z_grid, np.asarray(p["dep"], float),
                               np.asarray(p["val"], float), left=np.nan, right=np.nan)
    vals = vals[:, order]
    z = np.full((len(z_grid), len(x_grid)), np.nan)
    for i in range(len(z_grid)):
        finite = np.isfinite(vals[i])
        if finite.sum() < 2:
            continue
        z[i] = np.interp(x_grid, xs[finite], vals[i, finite], left=np.nan, right=np.nan)
    bd = np.asarray(bathy_dense, float)
    with np.errstate(invalid="ignore"):
        z[z_grid[:, None] > bd[None, :]] = np.nan
        z[:, ~(bd > 0.0)] = np.nan
    return z
