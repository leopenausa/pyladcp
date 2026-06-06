"""Reader for the cleaned CTD time-series `.cnv` (with merged GPS).

MORIA layout: whitespace-delimited, no header, 6 columns
``lat lon pressure elapsed_seconds temperature salinity`` at 1 Hz. Column roles are
parameterised via :class:`~ladcp.config.CastParams` for non-MORIA cruises.
"""

from __future__ import annotations

import numpy as np

from ..config import CastParams
from ..models import CTDTimeSeries


def read_clean_ctd(path: str, params: CastParams) -> CTDTimeSeries:
    data = np.loadtxt(path)
    if data.ndim == 1:
        data = data[None, :]
    ncol = data.shape[1]
    if ncol < params.ctd_fields_per_line:
        raise ValueError(
            f"{path}: expected >= {params.ctd_fields_per_line} columns, got {ncol}"
        )

    def col(field_1based: int) -> np.ndarray:
        return data[:, field_1based - 1]

    return CTDTimeSeries(
        time_elapsed_s=col(params.ctd_time_field),
        lat=col(params.nav_lat_field),
        lon=col(params.nav_lon_field),
        pressure=col(params.ctd_pressure_field),
        temperature=col(params.ctd_temperature_field),
        salinity=col(params.ctd_salinity_field),
        meta={"path": path, "n": data.shape[0]},
    )
