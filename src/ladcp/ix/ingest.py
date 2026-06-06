"""LADCP ingestion: raw PD0 (both heads) -> merged earth-velocity ensembles.

Faithful re-derivation of the dynamical part of ``loadrdi.m`` (and its ``updown``
subfunction): beam->earth transform (``b2earth``), percent-good / error-velocity /
vertical / horizontal velocity edits, up+down merge onto a common ensemble axis, and the
magnetic-declination rotation (``uvrot``). PD0 byte parsing is delegated to the proven
:mod:`ladcp.io.pd0` reader (kept from the previous build).

Combined-bin convention matches the MATLAB exactly: the merged stack is
``[flipud(up_bins); down_bins]`` so row 0 is the farthest up-looker bin and the last row
is the deepest down-looker bin. ``izu``/``izd`` give the combined-row indices of the
up/down blocks; ``izu[0]`` and ``izd[0]`` are the bins nearest each transducer.
"""

from __future__ import annotations

import warnings
from contextlib import contextmanager
from dataclasses import dataclass, field
from typing import Any

import numpy as np

from ..io.pd0 import read_pd0
from ..models import CoordFrame


@contextmanager
def _quiet_nan():
    """Suppress the all-NaN-slice / mean-of-empty warnings from surface/empty bins."""
    with warnings.catch_warnings():
        warnings.simplefilter("ignore", category=RuntimeWarning)
        yield


@dataclass
class LADCPData:
    """Merged dual-head LADCP ensembles (≈ legacy ``d`` after loadrdi).

    Velocity arrays are ``[nbin, nens]`` with ``nbin = nbin_up + nbin_down``; ``ru``/``rv``
    are declination-rotated earth east/north, ``rw`` vertical, ``re`` error. Times are
    ``datetime64[ns]`` UTC on the down-looker axis.
    """

    time: np.ndarray                 # [nens] datetime64
    ru: np.ndarray                   # [nbin, nens] east (rotated) m/s
    rv: np.ndarray                   # [nbin, nens] north (rotated) m/s
    rw: np.ndarray                   # [nbin, nens] vertical m/s
    re: np.ndarray                   # [nbin, nens] error m/s
    ts: np.ndarray                   # [nbin, nens] target strength (dB-counts)
    weight: np.ndarray               # [nbin, nens] correlation-based weight (normalised)
    zd: np.ndarray                   # [nbin_d] down bin distance from transducer (m)
    zu: np.ndarray                   # [nbin_u] up bin distance from transducer (m)
    izd: np.ndarray                  # combined-row indices of down block
    izu: np.ndarray                  # combined-row indices of up block
    pitch: np.ndarray                # [nens] deg (down head)
    roll: np.ndarray                 # [nens] deg
    heading: np.ndarray             # [nens] deg
    tilt: np.ndarray                 # [nens] deg
    temp: np.ndarray                 # [nens] deg C (down head)
    sv: np.ndarray                   # [nens] m/s instrument sound speed (down head)
    hbot: np.ndarray                 # [nens] RDI bottom-track range (m), down head
    bvel: np.ndarray                 # [nens, 4] bottom-track earth vel (u,v,w,e), rotated
    drot: float                      # magnetic declination applied (deg)
    meta: dict[str, Any] = field(default_factory=dict)

    @property
    def n_ens(self) -> int:
        return self.time.size

    @property
    def n_bin(self) -> int:
        return self.ru.shape[0]


# --------------------------------------------------------------------------- #
# Beam -> earth transform (loadrdi.m b2earth, vectorised over bins & ensembles)
# --------------------------------------------------------------------------- #
def b2earth(
    beam: np.ndarray,
    pitch: np.ndarray,
    roll: np.ndarray,
    head: np.ndarray,
    *,
    beams_up: bool,
    beamangle: float = 20.0,
    allow_3beam: bool = True,
) -> np.ndarray:
    """Convert RDI beam velocities to earth coordinates.

    ``beam`` is ``[4, ncell, nens]`` (beams 1-4). ``pitch``/``roll``/``head`` are ``[nens]``
    in degrees. Returns ``[4, ncell, nens]`` = (east, north, up, error). Three-beam
    solutions are reconstructed when exactly one beam is missing (``allow_3beam``);
    cells with >=2 missing beams become NaN. Mirrors the gimbal-pitch convention and the
    convex up/down beam combinations in ``loadrdi.m``.
    """
    d2r = np.pi / 180.0
    C30 = np.cos(beamangle * d2r)
    S30 = np.sin(beamangle * d2r)
    VXS = 1.0 / (2.0 * S30)
    VZS = 1.0 / (4.0 * C30)

    p = np.asarray(pitch, float) * d2r
    r = np.asarray(roll, float) * d2r
    h = np.asarray(head, float) * d2r
    RR = r
    KA = np.sqrt(1.0 - (np.sin(p) * np.sin(r)) ** 2)
    PP = np.arcsin(np.sin(p) * np.cos(r) / KA)
    CP, SP = np.cos(PP), np.sin(PP)
    CR, SR = np.cos(RR), np.sin(RR)
    CH, SH = np.cos(h), np.sin(h)  # each [nens]

    b = np.asarray(beam, float).copy()        # [4, ncell, nens]
    nan = np.isnan(b)
    nnan = nan.sum(axis=0)                     # [ncell, nens]
    one = nnan == 1
    if allow_3beam:
        b0, b1, b2, b3 = b[0], b[1], b[2], b[3]
        m0, m1, m2, m3 = nan[0], nan[1], nan[2], nan[3]
        r0, r1, r2, r3 = one & m0, one & m1, one & m2, one & m3
        b[0] = np.where(r0, _nz(-b1) + _nz(b2) + _nz(b3), b0)
        b[1] = np.where(r1, _nz(-b0) + _nz(b2) + _nz(b3), b[1])
        b[2] = np.where(r2, _nz(b0) + _nz(b1) - _nz(b3), b[2])
        b[3] = np.where(r3, _nz(b0) + _nz(b1) - _nz(b2), b[3])

    b0, b1, b2, b3 = b[0], b[1], b[2], b[3]
    if beams_up:
        VX = VXS * (-b0 + b1)
        VY = VXS * (-b2 + b3)
        VZ = VZS * (-(b0 + b1 + b2 + b3))
    else:
        VX = VXS * (b0 - b1)
        VY = VXS * (-b2 + b3)
        VZ = VZS * (b0 + b1 + b2 + b3)
    VE = VZS * (b0 + b1 - b2 - b3)

    VXE = VX * (CH * CR + SH * SR * SP) + VY * (SH * CP) + VZ * (CH * SR - SH * CR * SP)
    VYE = -VX * (SH * CR - CH * SR * SP) + VY * (CH * CP) - VZ * (SH * SR + CH * SP * CR)
    VZE = -VX * (SR * CP) + VY * SP + VZ * (CP * CR)

    valid = (nnan == 0) | (one if allow_3beam else False)
    out = np.full_like(b, np.nan)
    out[0] = np.where(valid, VXE, np.nan)
    out[1] = np.where(valid, VYE, np.nan)
    out[2] = np.where(valid, VZE, np.nan)
    out[3] = np.where(valid, VE, np.nan)
    return out


def _nz(x: np.ndarray) -> np.ndarray:
    """NaN->0 helper so reconstructed beams sum only finite partners."""
    return np.where(np.isfinite(x), x, 0.0)


def uvrot(u: np.ndarray, v: np.ndarray, rot_deg: float) -> tuple[np.ndarray, np.ndarray]:
    """Rotate (u,v) by ``rot_deg`` for magnetic declination (legacy ``uvrot.m``)."""
    rot = -rot_deg * np.pi / 180.0
    cr, sr = np.cos(rot), np.sin(rot)
    return u * cr - v * sr, u * sr + v * cr


# --------------------------------------------------------------------------- #
# Loader
# --------------------------------------------------------------------------- #
def load_ladcp(
    dn_path: str,
    up_path: str | None = None,
    *,
    drot: float,
    pglim: float = 0.0,
    elim: float = 0.5,
    vlim: float = 2.5,
    wlim: float = 0.2,
    tiltmax: tuple[float, float] = (22.0, 4.0),
    time_start: np.datetime64 | None = None,
    time_end: np.datetime64 | None = None,
    beamangle: float = 20.0,
) -> LADCPData:
    """Load + merge the two heads into earth-coordinate ensembles (loadrdi.m).

    ``drot`` is the magnetic declination (deg, east-positive). ``time_start``/``time_end``
    trim to the in-water cast (normally supplied by the CTD stage). Default edit limits
    match the GO-SHIP golden param struct.
    """
    dn = read_pd0(dn_path, head="dn")
    assert dn.coord_frame == CoordFrame.BEAM, "expected beam-coordinate PD0"
    earth_d = b2earth(dn.vel, dn.pitch, dn.roll, dn.heading,
                      beams_up=False, beamangle=beamangle)
    with _quiet_nan():
        ts_d = np.nanmedian(dn.echo, axis=0)       # [ncell_d, nens_d]
        cm_d = np.nanmedian(dn.corr, axis=0)
        hb_d = np.nanmedian(dn.bt_range, axis=0)   # [nens_d]
    bt_d = _bt_earth(dn, beamangle)            # [4, nens_d] earth bottom-track vel
    timd = dn.time

    if up_path is not None:
        up = read_pd0(up_path, head="up")
        earth_u = b2earth(up.vel, up.pitch, up.roll, up.heading,
                          beams_up=True, beamangle=beamangle)
        with _quiet_nan():
            ts_u = np.nanmedian(up.echo, axis=0)
            cm_u = np.nanmedian(up.corr, axis=0)
        timu = up.time
        # match up to down ensembles: start from nearest-time, then refine by an integer
        # W cross-correlation lag. The two heads' RTCs differ (~9 s on RB1606), so pure
        # time matching misaligns them during the fast downcast; loadrdi uses a W-based
        # bestlag for exactly this reason.
        tu = timu.astype("datetime64[ns]").astype(np.int64)
        td = timd.astype("datetime64[ns]").astype(np.int64)
        iu0 = np.abs(tu[None, :] - td[:, None]).argmin(axis=1)   # [nens_d]
        with _quiet_nan():
            wd = np.nanmedian(earth_d[2, :5, :], axis=0)         # down near-bin W
            wu0 = np.nanmedian(earth_u[2, :5, :], axis=0)[iu0]   # up W on down axis
        lag = _best_wlag(wd, wu0, maxlag=20)
        iu = np.clip(iu0 + lag, 0, earth_u.shape[2] - 1)
        nbin_u = earth_u.shape[1]
        # combined stack: flip up block (far->near becomes near at bottom of up block)
        def stack(eu, ed):
            return np.vstack([np.flipud(eu[:, iu]), ed])
        u = stack(earth_u[0], earth_d[0])
        v = stack(earth_u[1], earth_d[1])
        w = stack(earth_u[2], earth_d[2])
        e = stack(earth_u[3], earth_d[3])
        ts = np.vstack([np.flipud(ts_u[:, iu]), ts_d])
        cm = np.vstack([np.flipud(cm_u[:, iu]), cm_d])
        zu = up.cell_m * np.arange(nbin_u) + _first_bin_dist(up)
        zd = dn.cell_m * np.arange(earth_d.shape[1]) + _first_bin_dist(dn)
        # up block is flipped in the stack, so the nearest up bins are the LAST rows;
        # order izu near->far (== legacy fliplr(1:nbin_u)) so izu[:5] are nearest bins
        izu = np.flip(np.arange(nbin_u))
        izd = np.arange(earth_d.shape[1]) + nbin_u
    else:
        u, v, w, e = earth_d[0], earth_d[1], earth_d[2], earth_d[3]
        ts, cm = ts_d, cm_d
        zd = dn.cell_m * np.arange(earth_d.shape[1]) + _first_bin_dist(dn)
        zu = np.empty(0)
        izd = np.arange(earth_d.shape[1])
        izu = np.empty(0, dtype=int)

    # --- error-velocity edit (elim) -------------------------------------- #
    bad = np.abs(e) > elim
    u[bad] = v[bad] = w[bad] = np.nan

    # --- vertical-velocity edit (wlim): per-head reference from first 5 bins #
    wref = np.full_like(w, np.nan)
    with _quiet_nan():
        wref[izd] = np.nanmedian(w[izd[:5]], axis=0)
        if izu.size:
            wref[izu] = np.nanmedian(w[izu[:5]], axis=0)
    bad = np.abs(w - wref) > wlim
    u[bad] = v[bad] = w[bad] = np.nan

    # --- horizontal-speed edit (vlim) ------------------------------------ #
    with _quiet_nan():
        bad = np.sqrt(u**2 + v**2) > vlim
    u[bad] = v[bad] = w[bad] = np.nan

    # --- attitude (down head) + tilt ------------------------------------- #
    pit, rol, hdg = dn.pitch, dn.roll, dn.heading
    tilt = np.degrees(np.arcsin(np.sqrt(
        np.sin(np.radians(pit))**2 + np.sin(np.radians(rol))**2)))

    # --- time cut to in-water cast --------------------------------------- #
    if time_start is not None and time_end is not None:
        it = (timd >= time_start) & (timd <= time_end)
    else:
        it = np.isfinite(td := timd.astype("datetime64[ns]").astype(np.int64))
    sl = np.where(it)[0]

    # --- magnetic declination rotation ----------------------------------- #
    ru, rv = uvrot(u[:, sl], v[:, sl], drot)
    bu, bv = uvrot(bt_d[0, sl], bt_d[1, sl], drot)
    bvel = np.column_stack([bu, bv, bt_d[2, sl], bt_d[3, sl]])

    # --- correlation weight (normalised) --------------------------------- #
    weight = cm[:, sl].astype(float)
    with _quiet_nan():
        norm = np.nanmedian(np.nanmax(weight, axis=0))
    weight = weight / norm

    return LADCPData(
        time=timd[sl], ru=ru, rv=rv, rw=w[:, sl], re=e[:, sl],
        ts=ts[:, sl], weight=weight, zd=zd, zu=zu, izd=izd, izu=izu,
        pitch=pit[sl], roll=rol[sl], heading=hdg[sl], tilt=tilt[sl],
        temp=dn.temperature[sl], sv=dn.sound_speed[sl], hbot=hb_d[sl],
        bvel=bvel, drot=drot,
        meta={"dn": dn.meta, "n_ens_dn": dn.n_ens,
              "n_ens_up": (up.n_ens if up_path else 0),
              "freq_dn_khz": dn.freq_khz},
    )


def _best_wlag(wd: np.ndarray, wu: np.ndarray, maxlag: int = 20) -> int:
    """Integer lag (in ensembles) that best aligns up-W to down-W by correlation.

    Pragmatic stand-in for legacy ``bestlag``: maximises Pearson correlation of the two
    near-bin vertical-velocity series over shifts in [-maxlag, maxlag]. Robust for the
    constant-ping-rate GO-SHIP casts where the heads' clocks differ by a few ensembles.
    """
    best_lag, best_c = 0, -np.inf
    n = wd.size
    for lag in range(-maxlag, maxlag + 1):
        a = wd[max(0, -lag): n - max(0, lag)]
        b = wu[max(0, lag): n - max(0, -lag)]
        m = np.isfinite(a) & np.isfinite(b)
        if m.sum() < 50:
            continue
        c = np.corrcoef(a[m], b[m])[0, 1]
        if c > best_c:
            best_c, best_lag = c, lag
    return best_lag


def _bt_earth(dn, beamangle: float) -> np.ndarray:
    """Bottom-track beam velocities -> earth (one 'bin'). Returns [4, nens]."""
    if dn.bt_vel is None:
        return np.full((4, dn.n_ens), np.nan)
    velb = dn.bt_vel[:, None, :]      # [4, 1, nens]
    eb = b2earth(velb, dn.pitch, dn.roll, dn.heading,
                 beams_up=False, beamangle=beamangle)
    return eb[:, 0, :]


def _first_bin_dist(d) -> float:
    """Distance from transducer to centre of bin 1 (m).

    PD0 fixed-leader 'distance to first cell' is not surfaced by the reader; reconstruct
    from blank + cell as ``blank + cell`` (≈ the golden ``dist_*`` of ~8.2–8.4 m for
    8 m cells with zero blanking). Affects bin-depth geometry only, not W integration.
    """
    return float(d.blank_m + d.cell_m)
