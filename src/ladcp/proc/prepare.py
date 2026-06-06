"""Build the per-ensemble measurement arrays from the two heads (≈ legacy `loadrdi`).

For MORIA-05 the PD0 data are already in EARTH coordinates, so the 4 velocity
components are (east, north, up, error) — no beam→earth transform needed. We still
apply the magnetic-declination rotation (instrument compass is magnetic).

Combined bin axis follows the legacy convention ``[flip(uplooker), downlooker]``:
``izu`` indexes uplooker bins (shallower than the package), ``izd`` the downlooker.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any

import numpy as np

from ..config import CastParams
from ..models import CoordFrame, RawADCP
from .geometry import bin_distances


@dataclass
class Ensembles:
    """Combined dual-head per-ensemble arrays (legacy ``d``). Shapes [nbin, nt]."""

    time: np.ndarray                # [t] datetime64
    time_jul: np.ndarray            # [t] decimal days (for dt only)
    ru: np.ndarray                  # east water vel rel. package [nbin, t]
    rv: np.ndarray                  # north
    rw: np.ndarray                  # up (vertical)
    re: np.ndarray                  # error velocity
    weight: np.ndarray              # data weight (from correlation) [nbin, t]
    izd: np.ndarray                 # downlooker bin indices into the combined axis
    izu: np.ndarray                 # uplooker bin indices
    zd: np.ndarray                  # downlooker bin vertical distances (+down) [m]
    zu: np.ndarray                  # uplooker bin vertical distances (+up) [m]
    heading: np.ndarray             # [2, t] down/up head heading
    pitch: np.ndarray
    roll: np.ndarray
    tilt: np.ndarray                # [t] sqrt(pitch^2+roll^2)
    temp: np.ndarray                # [t] down head temperature
    sound_speed: np.ndarray         # [t] instrument sound speed (down head)
    bt_vel: np.ndarray | None       # [t, 4] bottom-track (e,n,u,err) rotated
    beam_angle: float = 20.0
    single_ping_err: float = np.nan
    pings_per_ens: int = 1
    meta: dict[str, Any] = field(default_factory=dict)

    @property
    def nbin(self) -> int:
        return self.ru.shape[0]

    @property
    def nt(self) -> int:
        return self.ru.shape[1]


def _rotate(u, v, deg):
    """Rotate (u,v) by ``deg`` degrees (east-from-north positive), legacy ``uvrot``."""
    a = np.deg2rad(deg)
    return u * np.cos(a) + v * np.sin(a), -u * np.sin(a) + v * np.cos(a)


def build_ensembles(master: RawADCP, slave: RawADCP | None,
                    params: CastParams, declination: float) -> Ensembles:
    if master.coord_frame != CoordFrame.EARTH:
        raise NotImplementedError(
            "beam-coordinate transform not yet implemented (MORIA-05 is earth coords)")

    # downlooker arrays on a common ensemble timeline (master is the time master)
    t = master.time
    nt = master.n_ens
    beam_angle = float(master.meta.get("beam_angle", 20.0))

    zd = bin_distances(master)           # +down distances [ncell]
    nd = zd.size

    # rotate earth velocities by declination (magnetic -> true)
    du, dv = _rotate(master.vel[0], master.vel[1], declination)
    dw, de = master.vel[2], master.vel[3]
    dwt = master.corr.mean(axis=0)       # correlation-based weight proxy [ncell, t]

    # percent-good editing (legacy loadrdi): RDI %good field 4 (4-beam solutions,
    # earth/WM mode) below p.pglim -> drop the measurement.
    du, dv, dw, de = _apply_pglim(du, dv, dw, de, master.pct_good, params.pglim)

    if slave is not None and params.up_dn_looker == 1:
        su, sv = _rotate(slave.vel[0], slave.vel[1], declination)
        sw_raw, se_raw = slave.vel[2], slave.vel[3]
        su, sv, sw_raw, se_raw = _apply_pglim(su, sv, sw_raw, se_raw,
                                              slave.pct_good, params.pglim)
        # interpolate slave onto master timeline (nearest ensemble)
        su, sv, sw, se, scorr = _match_slave(master, slave, su, sv, sw_raw, se_raw)
        zu = bin_distances(slave)
        nu = zu.size
        # combined order: [flip(uplooker), downlooker]
        ru = np.vstack([np.flipud(su), du])
        rv = np.vstack([np.flipud(sv), dv])
        rw = np.vstack([np.flipud(sw), dw])
        re = np.vstack([np.flipud(se), de])
        wt = np.vstack([np.flipud(scorr), dwt])
        izu = np.arange(nu)
        izd = np.arange(nu, nu + nd)
        head2 = np.vstack([master.heading, _interp_to(master, slave, slave.heading)])
        pit2 = np.vstack([master.pitch, _interp_to(master, slave, slave.pitch)])
        rol2 = np.vstack([master.roll, _interp_to(master, slave, slave.roll)])
    else:
        ru, rv, rw, re, wt = du, dv, dw, de, dwt
        zu = np.array([])
        izu = np.array([], dtype=int)
        izd = np.arange(nd)
        head2 = master.heading[None, :]
        pit2 = master.pitch[None, :]
        rol2 = master.roll[None, :]

    # normalise weight like legacy (corr / median(max corr))
    with np.errstate(invalid="ignore"):
        wt = wt / np.nanmedian(np.nanmax(wt, axis=0))
    # downweight bin 1 of each head
    if 1 in params.edit_mask_dn_bins:
        wt[izd[0]] *= params.weighbin1
    if izu.size and 1 in params.edit_mask_up_bins:
        wt[izu[-1]] *= params.weighbin1  # bin1 of uplooker = last in flipped order

    tilt = np.sqrt(master.pitch**2 + master.roll**2)

    # single-ping error from vertical-velocity coherence ACROSS bins (legacy):
    # std over bins per ensemble, then median over time, scaled by tan(beam angle).
    nmax = min(nd, 6)
    sw = np.nanstd(dw[1:nmax], axis=0)              # across bins, per ensemble
    spe = float(np.nanmedian(sw[np.isfinite(sw)]) / np.tan(np.deg2rad(beam_angle))) \
        if np.isfinite(sw).any() else np.nan

    bt = None
    if master.bt_vel is not None:
        bu, bv = _rotate(master.bt_vel[0], master.bt_vel[1], declination)
        bt = np.vstack([bu, bv, master.bt_vel[2], master.bt_vel[3]]).T  # [t,4]

    return Ensembles(
        time=t,
        time_jul=_to_juld(t),
        ru=ru, rv=rv, rw=rw, re=re, weight=wt,
        izd=izd, izu=izu, zd=zd, zu=zu,
        heading=head2, pitch=pit2, roll=rol2, tilt=tilt,
        temp=master.temperature, sound_speed=master.sound_speed,
        bt_vel=bt, beam_angle=beam_angle, single_ping_err=spe,
        pings_per_ens=master.pings_per_ens,
        meta={"declination": declination, "n_down": nd, "n_up": zu.size},
    )


def _apply_pglim(u, v, w, e, pct_good, pglim):
    """NaN out velocities whose %good field 4 is below ``pglim`` (legacy loadrdi)."""
    if not pglim or pct_good is None:
        return u, v, w, e
    pg4 = pct_good[3]                      # [ncell, t] % good 4-beam solutions
    bad = pg4 < pglim
    out = []
    for arr in (u, v, w, e):
        a = arr.copy()
        a[bad] = np.nan
        out.append(a)
    return tuple(out)


def _match_slave(master, slave, su, sv, sw, se):
    """Nearest-ensemble match of slave columns onto the master timeline."""
    idx = _nearest_idx(master.time, slave.time)
    return (su[:, idx], sv[:, idx], sw[:, idx], se[:, idx],
            slave.corr.mean(axis=0)[:, idx])


def _interp_to(master, slave, arr):
    return arr[_nearest_idx(master.time, slave.time)]


def _nearest_idx(t_master, t_slave):
    tm = t_master.astype("datetime64[ms]").astype("int64")
    ts = t_slave.astype("datetime64[ms]").astype("int64")
    j = np.searchsorted(ts, tm)
    j = np.clip(j, 1, len(ts) - 1)
    left = np.abs(tm - ts[j - 1]) < np.abs(tm - ts[j])
    j[left] -= 1
    return j


def _to_juld(t):
    ref = np.datetime64("0000-01-01")
    return (t - ref) / np.timedelta64(1, "D")
