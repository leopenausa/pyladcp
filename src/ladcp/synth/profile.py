"""Synthetic truth: ocean velocity profile, cast trajectory, and CTD time-series.

The truth profile is a barotropic mean ``(u0, v0)`` plus a *zero-mean* baroclinic
shear, so the depth-mean (barotropic) part is exactly what the velocity solve's
reference must recover, while the shear is what the relative profile must recover.
"""

from __future__ import annotations

from dataclasses import dataclass

import gsw
import numpy as np


@dataclass
class OceanTruth:
    """Known ocean velocity profile (the recovery-test target)."""

    z: np.ndarray   # depth [m, +down]
    u: np.ndarray   # east  [m/s]
    v: np.ndarray   # north [m/s]

    @property
    def ubar(self) -> float:
        return float(np.mean(self.u))

    @property
    def vbar(self) -> float:
        return float(np.mean(self.v))


def ocean_truth(*, depth: float = 1100.0, u0: float = -0.06, v0: float = 0.03,
                shear_amp: float = 0.18, shear_scale_m: float = 140.0,
                thermo_depth: float | None = None, n: int = 441) -> OceanTruth:
    """A realistic-but-recoverable baroclinic profile over the full water column.

    ``u`` is a **thermocline shear**: a ``tanh`` transition centred at ``thermo_depth``
    (default ``0.25*depth``) with e-folding width ``shear_scale_m`` — a near-uniform
    surface layer (``u0+shear_amp``) flowing over a counter-flowing deep layer
    (``u0-shear_amp``), the classic two-layer front. ``v`` is a broad half-cosine, so the
    velocity vector veers smoothly with depth. Both shear shapes are demeaned, so the
    depth means are exactly the barotropic ``(u0, v0)`` the inverse's reference must find.

    The shapes are deliberately distinct and low-wavenumber: a depth-*linear* shear in
    both components at once is the LADCP inverse's null space (it aliases with the package
    drifting over the down/up cast), and a mid-water bulge (a full sine) is heavily damped
    by the inverse smoothing. The ``tanh`` front + half-cosine pair recovers at corr ~1.
    """
    z = np.linspace(0.0, depth, n)
    z_th = thermo_depth if thermo_depth is not None else 0.25 * depth
    u_shear = np.tanh((z_th - z) / shear_scale_m)   # +1 above thermocline, -1 below
    v_shear = np.cos(np.pi * z / depth)             # broad half-cosine
    u = u0 + shear_amp * (u_shear - u_shear.mean())
    v = v0 + 0.7 * shear_amp * (v_shear - v_shear.mean())
    return OceanTruth(z=z, u=u, v=v)


def cast_trajectory(*, depth: float = 1080.0, descent_mps: float = 0.8,
                    ascent_mps: float = 0.8, ping_dt: float = 1.2,
                    surface_soak_s: float = 60.0, soak_depth: float = 5.0,
                    heave_amp: float = 0.0, heave_period_s: float = 30.0) -> dict:
    """Package-depth time-series for a down-cast then up-cast (plus a surface soak).

    Returns per-ping ``elapsed_s`` [s], ``z_pkg`` [m, +down], ``w_pkg`` [m/s, +down],
    and ``phase`` ("soak"/"down"/"up"). The fall/rise rate follows a smoothstep (slow
    start, fast middle, slow approach to the seabed -- a realistic cast), so the vertical
    velocity is a smooth, *non-repeating* hump. That unique gradient signature is what the
    ``bestlag`` clock-alignment locks onto: a constant fall rate gives a featureless boxcar
    ``w`` no gradient matcher can place, and a purely periodic heave aliases to any
    multiple of its period. A small heave (``heave_amp`` / ``heave_period_s``) adds the
    wire/ship texture a real package feels. ``w_pkg`` is the exact numerical derivative of
    ``z_pkg`` so the CTD and ADCP vertical velocities agree.
    """
    n_soak = max(1, int(round(surface_soak_s / ping_dt)))
    n_down = max(2, int(round((depth - soak_depth) / descent_mps / ping_dt)))
    n_up = max(2, int(round((depth - soak_depth) / ascent_mps / ping_dt)))

    def smoothstep(n: int) -> np.ndarray:
        f = np.linspace(0.0, 1.0, n)
        return 3.0 * f ** 2 - 2.0 * f ** 3          # 0->1, zero slope at both ends

    span = depth - soak_depth
    z_base = np.concatenate([
        np.full(n_soak, soak_depth),
        soak_depth + span * smoothstep(n_down),
        depth - span * smoothstep(n_up),
    ])
    elapsed = np.arange(z_base.size) * ping_dt
    heave = heave_amp * np.sin(2.0 * np.pi * elapsed / heave_period_s)
    z_pkg = np.clip(z_base + heave, 0.0, None)
    w_pkg = np.gradient(z_pkg, elapsed)
    phase = np.array(["soak"] * n_soak + ["down"] * n_down + ["up"] * n_up)
    return {"elapsed_s": elapsed, "z_pkg": z_pkg, "w_pkg": w_pkg, "phase": phase,
            "ping_dt": ping_dt}


def _temperature(z: np.ndarray) -> np.ndarray:
    """Plausible warm-surface / cold-deep profile [degC]."""
    return 12.0 - 9.0 * (1.0 - np.exp(-z / 400.0))


def _salinity(z: np.ndarray) -> np.ndarray:
    """Plausible salinity profile [PSU]."""
    return 35.0 + 0.3 * (1.0 - np.exp(-z / 500.0))


def ctd_series(traj: dict, *, lat0: float = 62.16, lon0: float = -11.53,
               ctd_dt: float = 1.0, seed: int = 0, noise: float = 0.0) -> dict:
    """1 Hz CTD time-series consistent with the cast trajectory.

    Pressure comes from the package depth via ``gsw.p_from_z``; T/S follow analytic
    profiles. The CTD time grid is independent of the ping grid (different ``dt``), so
    the down-stream ``synchronize`` cross-correlation has real work to do.
    """
    rng = np.random.default_rng(seed)
    total = float(traj["elapsed_s"][-1])
    elapsed = np.arange(0.0, total + ctd_dt, ctd_dt)
    z = np.interp(elapsed, traj["elapsed_s"], traj["z_pkg"])
    pressure = gsw.p_from_z(-z, lat0)
    temperature = _temperature(z) + (rng.normal(0, 0.01, z.size) if noise else 0.0)
    salinity = _salinity(z) + (rng.normal(0, 0.002, z.size) if noise else 0.0)
    # A small station-keeping GPS wander (~1 m) is always present, independent of `noise`:
    # a perfectly constant track makes the inverse's barotropic/GPS constraint degenerate
    # (0/0 -> NaN reference). 1e-5 deg is ~1.1 m, negligible for the barotropic estimate.
    lat = np.full(z.size, lat0) + rng.normal(0, 1e-5, z.size)
    lon = np.full(z.size, lon0) + rng.normal(0, 1e-5, z.size)
    return {"elapsed_s": elapsed, "lat": lat, "lon": lon, "pressure": pressure,
            "temperature": temperature, "salinity": salinity, "z_pkg": z}
