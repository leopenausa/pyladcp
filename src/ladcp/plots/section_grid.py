"""Bathymetry-aware objective-analysis gridding for cruise sections.

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
from scipy.ndimage import gaussian_filter


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


def smooth_z(z: np.ndarray, sigma: float) -> np.ndarray:
    """NaN-safe Gaussian smoothing (for contour overlays): fill, smooth, renormalise."""
    mask = np.isfinite(z)
    z_fill = np.where(mask, z, 0.0)
    z_sm = gaussian_filter(z_fill, sigma=sigma)
    w_sm = gaussian_filter(mask.astype(float), sigma=sigma)
    with np.errstate(invalid="ignore"):
        return np.where(w_sm > 0.01, z_sm / w_sm, np.nan)
