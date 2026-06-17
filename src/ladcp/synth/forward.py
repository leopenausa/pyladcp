"""Forward model: map the known ocean truth into per-cell PD0 fields for one head.

The ship holds station (package velocity over ground = 0), so the earth-frame velocity
the ADCP measures in a cell equals the ocean velocity at that cell's depth. The down
head samples ``z_pkg + bin_depth``; the up head samples ``z_pkg - bin_depth`` (its cells
sit above the package). Bottom-track velocity is zero (stationary package over a
stationary seabed), which gives the inverse a correct absolute reference, consistent with
the constant GPS track.
"""

from __future__ import annotations

import numpy as np

from ..models import CoordFrame, RawADCP
from .profile import OceanTruth

# bottom-track detection band (mirrors qa.bottom._BTRK_RANGE upper edge intent)
_BT_MIN_RANGE = 30.0


def synth_head(truth: OceanTruth, traj: dict, *, facing: str,
               t0: np.datetime64, seabed: float,
               freq_khz: int = 150, n_cells: int = 20, cell_m: float = 8.0,
               blank_m: float = 4.0, dist_first_m: float = 12.0,
               beam_angle_deg: int = 20, pings_per_ens: int = 1, wm_mode: int = 15,
               heading0: float = 200.0, compass_offset: float = 0.0,
               deck_pre_s: float = 240.0, deck_post_s: float = 240.0,
               seed: int = 0, noise: float = 0.0,
               echo_bg: float = 40.0, echo_amp: float = 80.0,
               echo_width: float = 8.0) -> RawADCP:
    """Forward-model one head (``facing`` = "down" | "up") into a :class:`RawADCP`.

    The ADCP record is padded with out-of-water "on deck" pings before and after the cast
    (``deck_pre_s`` / ``deck_post_s``), so the in-water descent/ascent vertical-velocity
    signature is a clear window for ``synchronize`` to lock the CTD onto -- as on a real
    deployment where the instrument pings before it enters the water.
    """
    rng = np.random.default_rng(seed + (0 if facing == "down" else 1000))
    ping_dt = float(traj["ping_dt"])
    n_pre = int(round(deck_pre_s / ping_dt))
    n_post = int(round(deck_post_s / ping_dt))
    z_cast = np.asarray(traj["z_pkg"], float)
    w_cast = np.asarray(traj["w_pkg"], float)
    # deck pings sit at the surface; flagged out-of-water so all returns are invalid
    z_pkg = np.concatenate([np.zeros(n_pre), z_cast, np.zeros(n_post)])
    w_pkg = np.concatenate([np.zeros(n_pre), w_cast, np.zeros(n_post)])
    in_water = np.concatenate([np.zeros(n_pre, bool), np.ones(z_cast.size, bool),
                               np.zeros(n_post, bool)])
    n_ens = z_pkg.size
    elapsed = np.arange(n_ens) * ping_dt
    time = t0 + (elapsed * 1000).astype("timedelta64[ms]")

    bin_depth = dist_first_m + np.arange(n_cells) * cell_m          # [n_cells]
    sign = 1.0 if facing == "down" else -1.0
    zc = z_pkg[None, :] + sign * bin_depth[:, None]                 # [n_cells, n_ens] ocean depth

    def at_depth(field: np.ndarray) -> np.ndarray:
        out = np.interp(zc.ravel(), truth.z, field).reshape(zc.shape)
        out[(zc < 0.0) | (zc > seabed)] = np.nan                   # outside the water column
        out[:, ~in_water] = np.nan                                 # on deck -> no valid returns
        return out

    u = at_depth(truth.u)
    v = at_depth(truth.v)
    valid = np.isfinite(u)
    if noise:
        u = u + rng.normal(0, noise, u.shape)
        v = v + rng.normal(0, noise, v.shape)

    # seabed return: cells at and just below the bed carry the seabed's horizontal Doppler
    # (= package velocity over ground = 0 for a station-keeping cast), not water velocity.
    # This is what the bottom-track estimator reads; the below-bottom edit keeps these cells
    # out of the water-velocity solve. Vertical W still tracks the package (below).
    if facing == "down":
        sea = (zc >= seabed) & (zc <= seabed + 2.0 * cell_m) & in_water[None, :]
        u = np.where(sea, 0.0, u)
        v = np.where(sea, 0.0, v)
        valid = valid | sea
    # vertical: package velocity (down +); a small per-cell jitter keeps the scatter floor finite
    w = np.broadcast_to(w_pkg[None, :], (n_cells, n_ens)).copy()
    w = w + rng.normal(0, max(noise, 0.01), w.shape)
    e = rng.normal(0, 0.005, (n_cells, n_ens))
    for a in (u, v, w, e):
        a[~valid] = np.nan

    vel = np.stack([u, v, w, e], axis=0)                           # [4, n_cells, n_ens]
    corr = np.where(valid, 120.0, 0.0)
    corr = np.broadcast_to(corr, (4, n_cells, n_ens)).copy()
    pct_good = np.broadcast_to(np.where(valid, 100.0, 0.0), (4, n_cells, n_ens)).copy()

    # echo amplitude: a range-decaying background (two-way transmission loss, realistic for
    # the raw dashboard) plus, on the down head, a constant-depth seabed reflector at range
    # seabed-z_pkg. The seabed detector locks on the depth-stacked reflector; the gentle TL
    # falloff just makes the raw echo look like real data.
    tl_db = 2.0 * (20.0 * np.log10(bin_depth) + 0.039 * bin_depth)   # two-way TL [dB] vs range
    bg_profile = echo_bg + 0.4 * (tl_db.max() - tl_db) / 0.45        # counts, falls with range
    ea = bg_profile[:, None] + rng.normal(0, 1.0, (n_cells, n_ens))
    if facing == "down":
        r_bot = seabed - z_pkg                                     # range to seabed per ping
        ea += echo_amp * np.exp(-0.5 * ((bin_depth[:, None] - r_bot[None, :]) / echo_width) ** 2)
    echo = np.broadcast_to(np.clip(ea, 0, 255), (4, n_cells, n_ens)).copy()

    # attitude / environment
    heading = np.full(n_ens, heading0 + compass_offset) + 0.5 * np.sin(elapsed / 120.0)
    pitch = 1.5 * np.sin(elapsed / 90.0)
    roll = 1.0 * np.cos(elapsed / 110.0)
    sound_speed = np.full(n_ens, 1500.0)
    temperature = np.interp(z_pkg, truth.z, 12.0 - 9.0 * (1 - np.exp(-truth.z / 400.0)))
    salinity = np.full(n_ens, 35.0)
    xmit_voltage = np.full(n_ens, 130.0)        # ~43 V battery estimate (healthy pack)

    # bottom track (down head only): range = seabed - z_pkg when within the profiling range,
    # velocity zero (stationary package over a stationary seabed)
    bt_range = bt_vel = None
    if facing == "down":
        max_range = bin_depth[-1]
        r_bot = seabed - z_pkg
        in_band = (r_bot >= _BT_MIN_RANGE) & (r_bot <= max_range)
        bt_range = np.where(in_band[None, :], r_bot[None, :], np.nan) * np.ones((4, 1))
        bt_vel = np.where(in_band[None, :], 0.0, np.nan) * np.ones((4, 1))
        bt_vel = bt_vel + np.where(np.isfinite(bt_vel), rng.normal(0, 0.003, bt_vel.shape), 0.0)

    return RawADCP(
        head=facing, coord_frame=CoordFrame.EARTH, facing=facing, freq_khz=freq_khz,
        n_beams=4, n_cells=n_cells, cell_m=cell_m, blank_m=blank_m, wm_mode=wm_mode,
        pings_per_ens=pings_per_ens, serial=99001 if facing == "down" else 99002,
        time=time, vel=vel, echo=echo, corr=corr, pct_good=pct_good,
        heading=heading, pitch=pitch, roll=roll, temperature=temperature,
        sound_speed=sound_speed, salinity=salinity, xmit_voltage=xmit_voltage,
        bt_range=bt_range, bt_vel=bt_vel,
        meta={"dist_first_m": dist_first_m, "beam_angle_deg": beam_angle_deg},
    )
