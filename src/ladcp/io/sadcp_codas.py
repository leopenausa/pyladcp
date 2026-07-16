"""Read CODAS-processed shipboard-ADCP NetCDF into a :class:`~.sadcp_vmdas.SadcpDataset`.

CODAS (Common Oceanographic Data Access System, UH/SOEST) post-processes the raw
VmDAS averages: editing (pflag), watertrack/bottomtrack calibration (rotation +
amplitude scaling) and smoothed navigation. Its "short form" NetCDF — written to
``<proc>/contour/<sonar>.nc`` by ``quick_adcp.py`` — already holds **absolute** ocean
velocity (``u``/``v``) plus position per ensemble, so unlike :mod:`.sadcp_vmdas` no
ship-velocity recovery is needed here; this reader only reshapes the file onto the
:class:`~.sadcp_vmdas.SadcpDataset` interface so the inverse constraint, Figure 9 and
the section plots work unchanged.

Mapping notes (verified on the MORIA OS150 STA product):

* ``time`` is "days since <yearbase>-01-01" stamped at the **end** of each average
  (raw STA start +~116 s); kept as-is — negligible inside the cast windows
  :func:`~.sadcp_vmdas.extract_profile` uses.
* ``depth`` is 2-D ``(time, depth_cell)`` but only wobbles by ~0.02 m with the sound
  speed; collapsed to the per-bin median (the grid the legacy ``mkSADCP`` assumed).
* ``pflag != 0`` marks edited cells (below bottom / bad PG / bad bin); they are blanked.

Calibration provenance (rotation phase/amplitude) lives in the CODAS processing dir
(``cal/rotate``), not in the NetCDF; the file's path recorded as ``sadcp_source``
(``codas:<path>``) is the pointer back to it.
"""

from __future__ import annotations

import re
from pathlib import Path

import numpy as np

from .sadcp_types import SadcpDataset


def resolve_codas_nc(path: str | Path) -> Path:
    """Resolve ``path`` to one CODAS NetCDF file.

    Accepts the ``.nc`` file itself, the processing directory (looks under
    ``contour/``), or any directory holding exactly one ``*.nc``.
    """
    p = Path(path)
    if p.is_file():
        return p
    if p.is_dir():
        for d in (p, p / "contour"):
            hits = sorted(d.glob("*.nc")) if d.is_dir() else []
            if len(hits) == 1:
                return hits[0]
            if len(hits) > 1:
                raise FileNotFoundError(
                    f"{d}: {len(hits)} .nc files ({', '.join(h.name for h in hits)}); "
                    "name the CODAS contour file explicitly")
    raise FileNotFoundError(f"no CODAS NetCDF at {path}")


def _decode_time(var) -> np.ndarray:
    """CODAS decimal-day time -> datetime64[ns] (units 'days since <epoch>')."""
    m = re.match(r"days\s+since\s+(\S+)(?:\s+(\S+))?", getattr(var, "units", ""))
    if not m:
        raise ValueError(f"unexpected CODAS time units: {getattr(var, 'units', None)!r}")
    epoch = np.datetime64(m.group(1) + ("T" + m.group(2) if m.group(2) else ""))
    days = np.asarray(var[:], dtype="float64")
    return (epoch.astype("datetime64[ns]")
            + (days * 86400e9).round().astype("timedelta64[ns]"))


def read_codas_nc(path: str | Path, *, pflag_max: int = 0,
                  pg_min: float = 0.0) -> SadcpDataset:
    """Read one CODAS short-form NetCDF into a :class:`~.sadcp_vmdas.SadcpDataset`.

    Cells whose editing flag exceeds ``pflag_max`` are blanked (default: keep only
    pflag 0, i.e. trust the CODAS editing). ``pg_min`` adds an optional percent-good
    floor on top (0 disables; CODAS' own editing already screens on PG).
    """
    from netCDF4 import Dataset

    nc_path = resolve_codas_nc(path)
    with Dataset(nc_path) as nc:
        time = _decode_time(nc.variables["time"])
        lat = np.ma.filled(nc.variables["lat"][:], np.nan).astype("float64")
        lon = np.ma.filled(nc.variables["lon"][:], np.nan).astype("float64")
        dep2 = np.ma.filled(nc.variables["depth"][:], np.nan)      # [t, z]
        u = np.ma.filled(nc.variables["u"][:], np.nan)             # [t, z] absolute
        v = np.ma.filled(nc.variables["v"][:], np.nan)
        bad = np.zeros(u.shape, bool)
        if "pflag" in nc.variables:
            bad |= np.ma.filled(nc.variables["pflag"][:], 7) > pflag_max
        if pg_min > 0 and "pg" in nc.variables:
            bad |= np.ma.filled(nc.variables["pg"][:], 0) < pg_min
        u[bad] = np.nan
        v[bad] = np.nan

        sonar = str(getattr(nc, "sonar", ""))
        m = re.search(r"(\d+)", sonar)
        freq_khz = int(m.group(1)) if m else 0
        if "transducer_depth" in nc.variables:
            xd = float(np.nanmedian(np.ma.filled(
                nc.variables["transducer_depth"][:], np.nan)))
        else:
            xd = np.nan

    depth = np.nanmedian(dep2, axis=0).astype("float64")
    if not (np.all(np.isfinite(depth)) and np.all(np.diff(depth) > 0)):
        raise ValueError(f"{nc_path}: depth grid not finite/monotonic after collapse")

    order = np.argsort(time)
    return SadcpDataset(
        time=time[order],
        lat=lat[order],
        lon=lon[order],
        depth=depth,
        u=np.ascontiguousarray(u[order].T.astype("float64")),
        v=np.ascontiguousarray(v[order].T.astype("float64")),
        freq_khz=freq_khz,
        transducer_depth=xd,
        file_type="CODAS",
        source=str(nc_path),
        n_files=1,
    )
