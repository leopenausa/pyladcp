"""Phase 5f — write the LDEO ``.lad`` / ``.bot`` velocity-profile files.

The ``.lad`` is the primary text deliverable: a short header (station / position /
declination) followed by ``z:u:v:ev`` columns, one row per depth bin. The ``.bot`` is the
bottom-track-referenced profile in the same layout (``z:u:v:err``). Both match the golden
``MORIA-80.{lad,bot}`` so they diff cleanly.
"""

from __future__ import annotations

from datetime import datetime

import numpy as np

from .inverse import BottomProfile, VelocityProfile


def _dm(value: float, pos: str, neg: str) -> str:
    """Format a decimal degree as ``D<hemi> MM.MMMM'`` (golden ``.lad`` convention)."""
    hemi = pos if value >= 0 else neg
    v = abs(value)
    deg = int(v)
    minutes = (v - deg) * 60.0
    return f"{deg:3d}°{hemi} {minutes:7.4f}'"


def write_lad(vp: VelocityProfile, path: str, *, station: str = "",
              lat: float = np.nan, lon: float = np.nan, drot: float = np.nan,
              time: datetime | None = None, filename: str | None = None) -> str:
    """Write a :class:`VelocityProfile` to an LDEO ``.lad`` file. Returns ``path``.

    ``lat``/``lon`` are decimal degrees (N/E positive); ``drot`` the magnetic declination.
    """
    t = time or datetime.utcnow()
    lines = [
        f"Filename    = {filename or station}",
        f"Date        = {t.year:04d}/{t.month:2d}/{t.day:2d}",
        f"Start_Time  = {t.hour:2d}:{t.minute:02d}:{t.second:02d}",
        f"Start_Lat   = {_dm(lat, 'N', 'S')}" if np.isfinite(lat) else "Start_Lat   = ",
        f"Start_Lon   = {_dm(lon, 'E', 'W')}" if np.isfinite(lon) else "Start_Lon   = ",
        f"Deviation   = {drot:.6f}" if np.isfinite(drot) else "Deviation   = ",
        "Columns     = z:u:v:ev",
    ]
    for z, u, v, ev in zip(vp.z, vp.u, vp.v, vp.uerr, strict=False):
        if not np.isfinite(u + v):
            continue
        e = ev if np.isfinite(ev) else 0.0
        lines.append(f"{z:6.1f} {u:6.3f} {v:6.3f} {e:6.3f} ")
    text = "\n".join(lines) + "\n"
    with open(path, "w") as fh:
        fh.write(text)
    return path


def write_bot(bp: BottomProfile, path: str, *, station: str = "",
              lat: float = np.nan, lon: float = np.nan, drot: float = np.nan,
              zbottom: float = np.nan, time: datetime | None = None,
              filename: str | None = None) -> str:
    """Write a :class:`BottomProfile` to an LDEO ``.bot`` file. Returns ``path``.

    Same header/column layout as :func:`write_lad` (golden ``MORIA-80.bot``), with an extra
    ``Bottom depth`` line and ``z:u:v:err`` columns.
    """
    t = time or datetime.utcnow()
    lines = [
        f"Filename    = {filename or station}",
        f"Date        = {t.year:04d}/{t.month:2d}/{t.day:2d}",
        f"Start_Time  = {t.hour:2d}:{t.minute:02d}:{t.second:02d}",
        f"Start_Lat   = {_dm(lat, 'N', 'S')}" if np.isfinite(lat) else "Start_Lat   = ",
        f"Start_Lon   = {_dm(lon, 'E', 'W')}" if np.isfinite(lon) else "Start_Lon   = ",
        f"Deviation   = {drot:.6f}" if np.isfinite(drot) else "Deviation   = ",
        f"Bottom depth= {int(zbottom)}" if np.isfinite(zbottom) else "Bottom depth= ",
        "Columns     = z:u:v:err",
    ]
    for z, u, v, e in zip(bp.z, bp.u, bp.v, bp.uerr, strict=False):
        if not np.isfinite(u + v):
            continue
        ev = e if np.isfinite(e) else 0.0
        lines.append(f"{z:6.1f} {u:6.3f} {v:6.3f} {ev:6.3f}")
    text = "\n".join(lines) + "\n"
    with open(path, "w") as fh:
        fh.write(text)
    return path
