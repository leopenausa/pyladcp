"""Beam-coordinate -> earth-coordinate velocity transform (port of LDEO_IX ``b2earth``).

RDI ADCPs can record single-ping velocities along each of the four slanted beams instead of
in earth coordinates. The LDEO_IX toolbox rotates them to earth at load time (``loadrdi.m``
calls ``b2earth`` per head for the water velocity and the four bottom-track beam velocities,
each with that head's own heading/pitch/roll). pyladcp previously had no such transform, so
beam-coordinate cruises could not be processed. This is a faithful, vectorized
port of ``b2earth`` (``loadrdi.m`` lines 1567-1768).

The resulting earth frame is *magnetic* ``(east, north, up, error)`` -- magnetic declination is
applied later in the velocity solve, exactly as for instruments that output earth coordinates
on board. The single-beam ``binremap`` step is hardwired off (legacy marks it buggy).
"""

from __future__ import annotations

import numpy as np


def beam_to_earth(vel: np.ndarray, heading: np.ndarray, pitch: np.ndarray, roll: np.ndarray, *,
                  beam_angle_deg: float, beams_up: bool, ignore_beam: int | None = None,
                  allow_3beam: bool = True, use_tilt: bool = True,
                  use_heading: bool = True) -> tuple[np.ndarray, int]:
    """Rotate per-beam radial velocities to earth coordinates (port of ``b2earth``).

    ``vel`` is ``[4, ncell, nt]`` (the four beam slots on axis 0); ``heading``/``pitch``/``roll``
    are ``[nt]`` deg. Returns ``(earth, n_3beam)`` where ``earth`` is ``[4, ncell, nt]`` =
    ``(east, north, up, error)`` (magnetic frame) and ``n_3beam`` is the number of (cell, ping)
    estimates recovered from a single missing beam via the error-velocity=0 closure.

    Reused for bottom track by passing the four BT beam velocities as ``bt_vel[:, None, :]``.
    """
    vel = np.asarray(vel, dtype=float)
    if vel.ndim != 3 or vel.shape[0] != 4:
        raise ValueError(f"vel must be [4, ncell, nt], got {vel.shape}")
    d2r = np.pi / 180.0
    c30 = np.cos(beam_angle_deg * d2r)
    s30 = np.sin(beam_angle_deg * d2r)
    # (legacy ZSG / SC depth-cell scaling is only used by the disabled binremap path; the up/down
    # distinction enters directly through the convex VX/VY/VZ branches below.)

    b = [vel[k].copy() for k in range(4)]      # each [ncell, nt]
    if ignore_beam is not None:
        b[ignore_beam - 1][:] = np.nan          # legacy ignore_beam is 1-based

    # --- 3-beam closure: reconstruct a single missing beam (error velocity = 0) ----------
    n_3beam = 0
    if allow_3beam:
        nan = [np.isnan(bk) for bk in b]
        n_nan = nan[0].astype(int) + nan[1] + nan[2] + nan[3]
        one = n_nan == 1
        n_3beam = int(one.sum())
        if n_3beam:
            # legacy l.1714-1722 (each branch reconstructs the missing beam from the other three)
            m0 = one & nan[0]
            m1 = one & nan[1]
            m2 = one & nan[2]
            m3 = one & nan[3]
            b[0][m0] = (-b[1][m0] + b[2][m0] + b[3][m0])
            b[1][m1] = (-b[0][m1] + b[2][m1] + b[3][m1])
            b[2][m2] = (b[0][m2] + b[1][m2] - b[3][m2])
            b[3][m3] = (b[0][m3] + b[1][m3] - b[2][m3])
    b1, b2, b3, b4 = b

    # --- instrument-frame components (convex up/down, legacy l.1732-1743) -----------------
    vxs = 1.0 / (2.0 * s30)
    vzs = 1.0 / (4.0 * c30)
    if beams_up:
        vx = vxs * (-b1 + b2)
        vy = vxs * (-b3 + b4)
        vz = vzs * (-b1 - b2 - b3 - b4)
    else:
        vx = vxs * (b1 - b2)
        vy = vxs * (-b3 + b4)
        vz = vzs * (b1 + b2 + b3 + b4)
    ve = vzs * (b1 + b2 - b3 - b4)              # error velocity (VES == VZS)

    # --- per-ensemble rotation factors (tilt-corrected pitch, legacy l.1637-1663) ---------
    roll_r = np.asarray(roll, float) * d2r
    pitch_r = np.asarray(pitch, float) * d2r
    head_r = np.asarray(heading, float) * d2r
    if use_tilt:
        ka = np.sqrt(1.0 - (np.sin(pitch_r) * np.sin(roll_r)) ** 2)
        pp = np.arcsin(np.sin(pitch_r) * np.cos(roll_r) / ka)
        cp, sp = np.cos(pp), np.sin(pp)
        cr, sr = np.cos(roll_r), np.sin(roll_r)
    else:
        cp = cr = np.ones_like(head_r)
        sp = sr = np.zeros_like(head_r)
    if use_heading:
        ch, sh = np.cos(head_r), np.sin(head_r)
    else:
        ch = np.ones_like(head_r)
        sh = np.zeros_like(head_r)

    # broadcast per-ensemble [nt] factors over cells [ncell, nt]
    ch, sh, cp, sp, cr, sr = (a[None, :] for a in (ch, sh, cp, sp, cr, sr))

    # --- earth rotation (legacy l.1750-1752) ----------------------------------------------
    vxe = vx * (ch * cr + sh * sr * sp) + vy * (sh * cp) + vz * (ch * sr - sh * cr * sp)
    vye = -vx * (sh * cr - ch * sr * sp) + vy * (ch * cp) - vz * (sh * sr + ch * sp * cr)
    vze = -vx * (sr * cp) + vy * sp + vz * (cp * cr)

    earth = np.stack([vxe, vye, vze, ve], axis=0)
    return earth, n_3beam
