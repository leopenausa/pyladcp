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

# Seabird .cnv column names -> our roles, matched as a prefix on the ``# name N = <var>``
# token. Lets the reader auto-map a headered .cnv from a new cruise without a hand-set field
# map; the headerless MORIA clean files fall through to the ``params`` field numbers.
_SBE_ROLES: dict[str, tuple[str, ...]] = {
    "pressure": ("prdm", "prsm", "pr", "depsm", "depth"),
    "temperature": ("t090c", "t190c", "tv290c", "t068c", "t090", "t168c", "tnc90c"),
    "salinity": ("sal00", "sal11"),
    "lat": ("latitude", "lat"),
    "lon": ("longitude", "lon"),
    "time": ("times", "timej", "timeq", "timek", "timen"),
}


def _parse_cnv_header(path: str) -> tuple[int, dict[str, int]]:
    """Return ``(n_header_lines, role->0based-col)`` from a Seabird ``.cnv`` header.

    Reads the leading ``#``/``*`` comment block, mapping each ``# name N = <var>: ...`` line
    to a role via :data:`_SBE_ROLES`. Empty role map (and 0 header lines) for a headerless
    file. The first matching column wins for each role.
    """
    roles: dict[str, int] = {}
    n_header = 0
    with open(path, encoding="latin-1") as fh:
        for line in fh:
            s = line.strip()
            if not (s.startswith("#") or s.startswith("*")):
                break
            n_header += 1
            body = s.lstrip("#* ").lower()
            if not body.startswith("name "):
                continue
            # "name 2 = t090C: Temperature [ITS-90, deg C]"
            try:
                idx_part, var_part = body[5:].split("=", 1)
                col = int(idx_part.strip())
                var = var_part.strip().split(":", 1)[0].strip()
            except (ValueError, IndexError):
                continue
            for role, prefixes in _SBE_ROLES.items():
                if role not in roles and any(var.startswith(pre) for pre in prefixes):
                    roles[role] = col
                    break
    return n_header, roles


def read_ctd_cnv(path: str, params: CastParams | None = None) -> CTDTimeSeries:
    """Read a cleaned CTD ``.cnv`` into a :class:`CTDTimeSeries`.

    Column positions are resolved data-first (#4a): if the file carries a Seabird header,
    the ``# name N = <var>`` lines auto-map the columns; any role the header does not cover
    falls back to the ``params`` field number (1-based, like the legacy loader). Headerless
    files (the MORIA clean ``.cnv``) use the ``params`` map throughout.
    """
    p = params or CastParams(station="")
    n_header, roles = _parse_cnv_header(path)

    def field(role: str, fallback_1based: int) -> int:
        """0-based column for ``role``: header-derived if present, else ``params``."""
        return roles[role] if role in roles else fallback_1based - 1

    pres = field("pressure", p.ctd_pressure_field)
    temp = field("temperature", p.ctd_temperature_field)
    salt = field("salinity", p.ctd_salinity_field)
    lat = field("lat", p.nav_lat_field)
    lon = field("lon", p.nav_lon_field)
    tcol = field("time", p.ctd_time_field)

    rows = np.loadtxt(path, dtype=float, skiprows=n_header)
    need = max(pres, temp, salt, lat, lon, tcol) + 1
    if rows.ndim != 2 or rows.shape[1] < need:
        raise ValueError(
            f"{path}: expected >= {need} columns, got "
            f"{rows.shape[1] if rows.ndim == 2 else rows.shape}"
        )

    return CTDTimeSeries(
        time_elapsed_s=rows[:, tcol],
        lat=rows[:, lat],
        lon=rows[:, lon],
        pressure=rows[:, pres],
        temperature=rows[:, temp],
        salinity=rows[:, salt],
        meta={"path": path, "n_scans": rows.shape[0],
              "header_lines": n_header, "header_roles": roles},
    )
