"""Reader for the *cleaned* CTD ``.cnv`` files used as LADCP nav + time-series input.

These are the SBE-processed, despiked, 1 s bin-averaged files in ``New_golden/Good/CTD``.
They carry **no header** (``header_lines = 0``) and are plain whitespace columns. The
column map matches the legacy ``f`` struct recorded in ``MORIA-80.mat``::

    1: lat [deg]   2: lon [deg]   3: pressure [dbar]
    4: time [s, elapsed since cast start]   5: temperature [degC]   6: salinity [PSU]

The surface-soak portion shows pressure bobbing a few dbar (swell); that is real and is
left untouched here — trimming/water-entry detection is a downstream QA concern.
"""

from __future__ import annotations

import numpy as np

from ..config import CastParams
from ..models import CTDTimeSeries


def read_ctd_cnv(path: str, params: CastParams | None = None) -> CTDTimeSeries:
    """Read a cleaned CTD ``.cnv`` into a :class:`CTDTimeSeries`.

    Field positions follow ``params`` (1-based, like the legacy loader) so the same
    reader works if a cruise reorders columns. Defaults match the MORIA clean files.
    """
    p = params or CastParams(station="")
    rows = np.loadtxt(path, dtype=float)
    if rows.ndim != 2 or rows.shape[1] < max(
        params_fields := (
            p.nav_lat_field, p.nav_lon_field, p.ctd_pressure_field,
            p.ctd_time_field, p.ctd_temperature_field, p.ctd_salinity_field,
        )
    ):
        raise ValueError(
            f"{path}: expected >= {max(params_fields)} columns, got "
            f"{rows.shape[1] if rows.ndim == 2 else rows.shape}"
        )

    def col(field_1based: int) -> np.ndarray:
        return rows[:, field_1based - 1]

    return CTDTimeSeries(
        time_elapsed_s=col(p.ctd_time_field),
        lat=col(p.nav_lat_field),
        lon=col(p.nav_lon_field),
        pressure=col(p.ctd_pressure_field),
        temperature=col(p.ctd_temperature_field),
        salinity=col(p.ctd_salinity_field),
        meta={"path": path, "n_scans": rows.shape[0]},
    )
