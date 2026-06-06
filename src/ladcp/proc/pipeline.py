"""End-to-end single-cast processing: raw + CTD -> ProfileResult.

Orchestrates the stages (build ensembles, depth/time-match, bottom detection, edit,
super-ensembles, ship drift, inverse solve). This is the Python equivalent of the
LDEO_IX ``process_cast`` path for a clean dual-head, earth-coordinate cast.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np

from ..config import CastParams
from ..models import CTDTimeSeries, ProfileResult, RawADCP
from .depth import solve_depth
from .inversion import edit, form_superensembles, invert
from .prepare import build_ensembles


@dataclass
class CastResult:
    profile: ProfileResult
    diagnostics: dict


def process_cast(master: RawADCP, slave: RawADCP | None, ctd: CTDTimeSeries,
                 params: CastParams, declination: float) -> CastResult:
    ens = build_ensembles(master, slave, params, declination)
    depth = solve_depth(ens, ctd)

    zbottom = _detect_bottom(master, ens, depth, declination)

    edit(ens, depth, params, zbottom)
    se = form_superensembles(ens, depth, params, ctd)

    uship, lat0, lon0 = _ship_drift(ctd, depth)
    # total in-water cast duration (legacy p.dt_profile) for the barotropic weight
    dt_profile = float(ctd.time_elapsed_s[-1] - ctd.time_elapsed_s[0])

    pr = invert(se, params, lat=lat0, lon=lon0, uship=uship,
                zbottom=zbottom, station=params.station, declination=declination,
                dt_profile=dt_profile)
    pr.date = ens.time[ens.time.size // 2]

    diag = {
        "lag_seconds": depth.lag_seconds,
        "max_depth": depth.max_depth,
        "zbottom": zbottom,
        "n_superens": se.ru.shape[1],
        "velerr": pr.config.get("velerr"),
        "uship": uship,
    }
    return CastResult(profile=pr, diagnostics=diag)


def _detect_bottom(master: RawADCP, ens, depth, declination) -> float | None:
    """Bottom depth = package depth + height-above-bottom near the deepest point."""
    if master.bt_range is None:
        return None
    hbot = np.nanmean(master.bt_range, axis=0)        # vertical distance to bottom [m]
    zpos = -depth.z                                    # positive down
    near = zpos > (depth.max_depth - 200)
    ok = near & np.isfinite(hbot) & (hbot > 0)
    if ok.sum() < 5:
        return None
    zb = np.nanmedian(zpos[ok] + hbot[ok])
    # reject if implausible vs deepest CTD depth
    if zb < depth.max_depth - 20:
        return None
    return float(zb)


def _ship_drift(ctd: CTDTimeSeries, depth):
    iw = depth.in_water
    lat = ctd.lat
    lon = ctd.lon
    # use first/last valid CTD fixes over the cast elapsed time
    t = ctd.time_elapsed_s.astype(float)
    dt = t[-1] - t[0]
    lat0, lat1 = lat[0], lat[-1]
    lon0, lon1 = lon[0], lon[-1]
    latm = 0.5 * (lat0 + lat1)
    xdisp = (lon1 - lon0) * np.cos(np.deg2rad(latm)) * 60.0 * 1852.0
    ydisp = (lat1 - lat0) * 60.0 * 1852.0
    uship = (xdisp + 1j * ydisp) / dt if dt > 0 else np.nan
    return uship, float(latm), float(0.5 * (lon0 + lon1))
