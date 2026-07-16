"""Pure tidy-table builders — the single source of column/unit truth for every export.

Each ``*_frame`` turns part of a :class:`~ladcp.qa.inverse.VelocityResult` (plus the QA
:class:`~ladcp.models.QCMetrics`) into a :class:`pandas.DataFrame` with stable, documented
column names; :func:`metadata_dict` and :func:`summary_row` give the per-station scalars.
No file I/O lives here, so the formatting in :mod:`~ladcp.export.odv`, ``.netcdf`` and
``.xlsx`` all read from one consistent shape and the builders are trivially testable.
"""

from __future__ import annotations

from dataclasses import dataclass
from datetime import datetime

import numpy as np
import pandas as pd

from ..models import QCMetrics
from ..qa.inverse import VelocityResult


@dataclass
class StationExport:
    """One station's solved velocity + QA, with the scalars the writers need.

    ``time`` is the cast start (UTC); ``drot`` the magnetic declination applied. SADCP
    provenance is carried for the metadata only (``None`` when no ship-ADCP constraint).
    """

    station: str
    cruise: str
    lat: float
    lon: float
    time: datetime
    drot: float
    solver: str
    result: VelocityResult
    qc: QCMetrics
    sadcp_source: str | None = None
    sadcp_freq_khz: float | None = None


# Column units, kept here so every format reports the same thing.
UNITS = {
    "depth_m": "m",
    "u_ms": "m s-1",
    "v_ms": "m s-1",
    "uerr_ms": "m s-1",
    "verr_ms": "m s-1",
    "u_shear_1s": "s-1",
    "v_shear_1s": "s-1",
    "n_vel": "1",
    "n": "1",
}
LONG_NAME = {
    "depth_m": "depth below sea surface",
    "u_ms": "eastward sea water velocity",
    "v_ms": "northward sea water velocity",
    "uerr_ms": "formal velocity uncertainty (inverse covariance, not a full error budget)",
    "verr_ms": "formal velocity uncertainty (inverse covariance, not a full error budget)",
    "u_shear_1s": "eastward velocity shear",
    "v_shear_1s": "northward velocity shear",
    "n_vel": "velocity samples per bin",
    "n": "samples per bin",
}
# CF standard names, so CF-aware tools (ODV, QGIS, Ferret) can interpret the columns.
STANDARD_NAME = {
    "depth_m": "depth",
    "u_ms": "eastward_sea_water_velocity",
    "v_ms": "northward_sea_water_velocity",
    # CF "standard_error" name modifier: uerr/verr are the standard error of the velocity.
    "uerr_ms": "eastward_sea_water_velocity standard_error",
    "verr_ms": "northward_sea_water_velocity standard_error",
}


def pyladcp_version() -> str:
    from importlib.metadata import PackageNotFoundError, version

    try:
        return version("pyladcp")
    except PackageNotFoundError:
        return "unknown"


def profile_frame(result: VelocityResult) -> pd.DataFrame:
    """Absolute velocity profile (the ``.lad`` deliverable). Drops all-NaN velocity bins."""
    vp = result.vp
    df = pd.DataFrame({
        "depth_m": np.asarray(vp.z, float),
        "u_ms": np.asarray(vp.u, float),
        "v_ms": np.asarray(vp.v, float),
        "uerr_ms": np.asarray(vp.uerr, float),
        "n_vel": np.asarray(vp.nvel, float),
    })
    keep = np.isfinite(df["u_ms"]) & np.isfinite(df["v_ms"])   # match write_lad's row filter
    return df.loc[keep].reset_index(drop=True)


def bottomtrack_frame(result: VelocityResult) -> pd.DataFrame | None:
    """Bottom-track-referenced profile (the ``.bot`` deliverable), or ``None`` if absent."""
    bp = result.bp
    if bp is None or bp.n_bins == 0:
        return None
    df = pd.DataFrame({
        "depth_m": np.asarray(bp.z, float),
        "u_ms": np.asarray(bp.u, float),
        "v_ms": np.asarray(bp.v, float),
        "uerr_ms": np.asarray(bp.uerr, float),
        "n": np.asarray(bp.n, float),
    })
    keep = np.isfinite(df["u_ms"]) & np.isfinite(df["v_ms"])
    return df.loc[keep].reset_index(drop=True)


def shear_frame(result: VelocityResult) -> pd.DataFrame:
    """Baroclinic shear profile (true frame)."""
    sh = result.shear
    return pd.DataFrame({
        "depth_m": np.asarray(sh.z, float),
        "u_ms": np.asarray(sh.u, float),
        "v_ms": np.asarray(sh.v, float),
        "u_shear_1s": np.asarray(sh.u_shear, float),
        "v_shear_1s": np.asarray(sh.v_shear, float),
        "n": np.asarray(sh.n, float),
    })


def sadcp_frame(result: VelocityResult) -> pd.DataFrame | None:
    """Ship-ADCP constraint profile ``[z, u, v, verr]``, or ``None`` if not used."""
    sv = result.sadcp
    if sv is None or len(sv) == 0:
        return None
    sv = np.asarray(sv, float)
    return pd.DataFrame({
        "depth_m": sv[:, 0],
        "u_ms": sv[:, 1],
        "v_ms": sv[:, 2],
        "verr_ms": sv[:, 3],
    })


def locate_frame(df: pd.DataFrame, export: StationExport) -> pd.DataFrame:
    """Prepend station id + position columns so an extracted data sheet is self-locating."""
    out = df.copy()
    out.insert(0, "station", export.station)
    out.insert(1, "latitude_deg", float(export.lat))
    out.insert(2, "longitude_deg", float(export.lon))
    return out


def qa_frame(qc: QCMetrics) -> pd.DataFrame:
    """The QA scorecard as one row per diagnostic."""
    rows = []
    for name, m in qc.metrics.items():
        rows.append({
            "metric": name,
            "value": m.value,
            "unit": m.unit,
            "status": m.status.value,
            "threshold": m.threshold,
            "note": m.note,
        })
    return pd.DataFrame(rows, columns=["metric", "value", "unit", "status",
                                       "threshold", "note"])


def metadata_dict(export: StationExport) -> dict:
    """Flat scalar metadata for headers / NetCDF global attrs."""
    r = export.result
    md = {
        "station": export.station,
        "cruise": export.cruise,
        "latitude_deg": float(export.lat),
        "longitude_deg": float(export.lon),
        "time_utc": export.time.isoformat() if export.time is not None else "",
        "magnetic_declination_deg": float(export.drot),
        "bottom_depth_m": float(r.zbottom),
        "ubar_ms": float(r.vp.ubar),
        "vbar_ms": float(r.vp.vbar),
        "solver": export.solver,
        "n_bins": int(np.isfinite(r.vp.u).sum()),
        "overall_status": export.qc.overall_status.value,
        "pyladcp_version": pyladcp_version(),
    }
    if export.sadcp_source is not None:
        md["sadcp_source"] = export.sadcp_source
    if export.sadcp_freq_khz is not None:
        md["sadcp_freq_khz"] = float(export.sadcp_freq_khz)
    return md


def summary_row(export: StationExport) -> dict:
    """One row for the cruise summary CSV."""
    r = export.result
    return {
        "station": export.station,
        "latitude_deg": float(export.lat),
        "longitude_deg": float(export.lon),
        "time_utc": export.time.isoformat() if export.time is not None else "",
        "overall_status": export.qc.overall_status.value,
        "ubar_ms": float(r.vp.ubar),
        "vbar_ms": float(r.vp.vbar),
        "bottom_depth_m": float(r.zbottom),
        "n_bins": int(np.isfinite(r.vp.u).sum()),
        "sadcp_independent_rms_ms": (float(r.sadcp_independent_rms)
                                     if r.sadcp_independent_rms is not None
                                     else float("nan")),
    }
