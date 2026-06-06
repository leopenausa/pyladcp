"""Magnetic declination via the IGRF model (replaces legacy ``magdev.m``/IGRF*.COF).

The legacy MORIA processing computed declination from the cast position and date when
``p.drot`` was NaN. We reproduce that with the IGRF-13 coefficients (``ppigrf``).
Declination D = atan2(B_east, B_north).
"""

from __future__ import annotations

from datetime import datetime

import numpy as np


def magnetic_declination(lat: float, lon: float, when: datetime, alt_km: float = 0.0) -> float:
    """Return magnetic declination [deg], east positive, at (lat, lon, date)."""
    import ppigrf

    if when.tzinfo is not None:
        when = when.replace(tzinfo=None)   # ppigrf compares against tz-naive coefficients
    be, bn, _bu = ppigrf.igrf(lon, lat, alt_km, when)
    d = np.degrees(np.arctan2(np.asarray(be).ravel()[0], np.asarray(bn).ravel()[0]))
    return float(d)
