"""CF-style NetCDF writers (per-station and cruise) via xarray.

A station file is one flat :class:`xarray.Dataset`: because the profile, bottom-track,
shear and SADCP arrays live on different depth grids, each gets its own depth coordinate
(``depth``, ``depth_bt``, ``depth_shear``, ``depth_sadcp``) and its variables hang off it.
Global attributes carry the station metadata. The cruise file stacks every station's
*absolute* profile onto a common union depth grid (NaN-padded) along a ``station`` dimension.
"""

from __future__ import annotations

import numpy as np
import pandas as pd
import xarray as xr

from .tables import (
    LONG_NAME,
    STANDARD_NAME,
    UNITS,
    StationExport,
    bottomtrack_frame,
    metadata_dict,
    profile_frame,
    sadcp_frame,
    shear_frame,
)


def _var(values, dim: str, col: str) -> xr.DataArray:
    attrs = {}
    if col in UNITS:
        attrs["units"] = UNITS[col]
    if col in LONG_NAME:
        attrs["long_name"] = LONG_NAME[col]
    if col in STANDARD_NAME:
        attrs["standard_name"] = STANDARD_NAME[col]
    return xr.DataArray(np.asarray(values, float), dims=(dim,), attrs=attrs)


def _depth_coord(values, name: str) -> dict:
    return {name: xr.DataArray(np.asarray(values, float), dims=(name,),
                               attrs={"units": "m", "long_name": "depth below sea surface",
                                      "standard_name": "depth", "positive": "down",
                                      "axis": "Z"})}


def _scalar_coords(export: StationExport) -> dict:
    """Scalar CF latitude/longitude (and time) coordinates that locate the station."""
    coords = {
        "latitude": xr.DataArray(
            float(export.lat),
            attrs={"units": "degrees_north", "standard_name": "latitude",
                   "long_name": "station latitude", "axis": "Y"}),
        "longitude": xr.DataArray(
            float(export.lon),
            attrs={"units": "degrees_east", "standard_name": "longitude",
                   "long_name": "station longitude", "axis": "X"}),
    }
    if export.time is not None:
        coords["time"] = xr.DataArray(
            np.datetime64(export.time),
            attrs={"standard_name": "time", "long_name": "cast start time", "axis": "T"})
    return coords


def _station_dataset(export: StationExport) -> xr.Dataset:
    prof = profile_frame(export.result)
    coords = _depth_coord(prof["depth_m"], "depth")
    coords |= _scalar_coords(export)
    data = {
        "u": _var(prof["u_ms"], "depth", "u_ms"),
        "v": _var(prof["v_ms"], "depth", "v_ms"),
        "uerr": _var(prof["uerr_ms"], "depth", "uerr_ms"),
        "nvel": _var(prof["n_vel"], "depth", "n_vel"),
    }
    # CF ancillary_variables: point u/v at their standard-error variable (uerr).
    data["u"].attrs["ancillary_variables"] = "uerr"
    data["v"].attrs["ancillary_variables"] = "uerr"

    bt = bottomtrack_frame(export.result)
    if bt is not None:
        coords |= _depth_coord(bt["depth_m"], "depth_bt")
        data |= {
            "u_bt": _var(bt["u_ms"], "depth_bt", "u_ms"),
            "v_bt": _var(bt["v_ms"], "depth_bt", "v_ms"),
            "uerr_bt": _var(bt["uerr_ms"], "depth_bt", "uerr_ms"),
        }
        data["u_bt"].attrs["ancillary_variables"] = "uerr_bt"
        data["v_bt"].attrs["ancillary_variables"] = "uerr_bt"

    sh = shear_frame(export.result)
    coords |= _depth_coord(sh["depth_m"], "depth_shear")
    data |= {
        "u_shear": _var(sh["u_ms"], "depth_shear", "u_ms"),
        "v_shear": _var(sh["v_ms"], "depth_shear", "v_ms"),
    }

    sv = sadcp_frame(export.result)
    if sv is not None:
        coords |= _depth_coord(sv["depth_m"], "depth_sadcp")
        data |= {
            "u_sadcp": _var(sv["u_ms"], "depth_sadcp", "u_ms"),
            "v_sadcp": _var(sv["v_ms"], "depth_sadcp", "v_ms"),
            "verr_sadcp": _var(sv["verr_ms"], "depth_sadcp", "verr_ms"),
        }

    return xr.Dataset(data, coords=coords, attrs=_global_attrs(export))


def _global_attrs(export: StationExport) -> dict:
    md = metadata_dict(export)
    md["Conventions"] = "CF-1.8"
    md["title"] = f"LADCP velocity profile — {export.station}"
    return md


def write_station_nc(export: StationExport, path: str) -> str:
    """Write one station's solution as a CF-style NetCDF profile. Returns ``path``."""
    _station_dataset(export).to_netcdf(path)
    return path


def write_cruise_nc(exports: list[StationExport], path: str, *, cruise: str = "") -> str:
    """Stack every station's absolute profile on a common depth grid. Returns ``path``."""
    frames = [(e, profile_frame(e.result)) for e in exports]
    union = np.unique(np.concatenate(
        [f["depth_m"].to_numpy() for _, f in frames if len(f)] or [np.array([])]))

    stations = [e.station for e, _ in frames]
    nz, ns = union.size, len(frames)
    u = np.full((ns, nz), np.nan)
    v = np.full((ns, nz), np.nan)
    uerr = np.full((ns, nz), np.nan)
    for i, (_, f) in enumerate(frames):
        if not len(f):
            continue
        idx = np.searchsorted(union, f["depth_m"].to_numpy())
        u[i, idx] = f["u_ms"].to_numpy()
        v[i, idx] = f["v_ms"].to_numpy()
        uerr[i, idx] = f["uerr_ms"].to_numpy()

    md = [metadata_dict(e) for e, _ in frames]
    time = pd.to_datetime([m["time_utc"] or None for m in md])
    ds = xr.Dataset(
        {
            "u": (("station", "depth"), u, {"units": "m s-1",
                                            "long_name": "eastward sea water velocity",
                                            "standard_name": "eastward_sea_water_velocity",
                                            "ancillary_variables": "uerr"}),
            "v": (("station", "depth"), v, {"units": "m s-1",
                                            "long_name": "northward sea water velocity",
                                            "standard_name": "northward_sea_water_velocity",
                                            "ancillary_variables": "uerr"}),
            "uerr": (("station", "depth"), uerr,
                     {"units": "m s-1",
                      "long_name": "formal velocity uncertainty (inverse covariance, "
                                   "not a full error budget)",
                      "standard_name": "eastward_sea_water_velocity standard_error"}),
            "bottom_depth": ("station", [m["bottom_depth_m"] for m in md],
                             {"units": "m", "long_name": "bottom depth"}),
        },
        coords={
            "station": stations,
            "depth": xr.DataArray(union, dims=("depth",),
                                  attrs={"units": "m", "positive": "down", "axis": "Z",
                                         "standard_name": "depth",
                                         "long_name": "depth below sea surface"}),
            "latitude": ("station", [m["latitude_deg"] for m in md],
                         {"units": "degrees_north", "standard_name": "latitude",
                          "long_name": "station latitude", "axis": "Y"}),
            "longitude": ("station", [m["longitude_deg"] for m in md],
                          {"units": "degrees_east", "standard_name": "longitude",
                           "long_name": "station longitude", "axis": "X"}),
            "time": ("station", time,
                     {"standard_name": "time", "long_name": "cast start time", "axis": "T"}),
        },
        attrs={"Conventions": "CF-1.8", "cruise": cruise,
               "title": f"LADCP cruise velocity sections — {cruise}"},
    )
    ds.to_netcdf(path)
    return path
