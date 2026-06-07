"""Roadmap #3 — CF-style NetCDF writers (per-station + cruise)."""

from __future__ import annotations

import numpy as np
import xarray as xr

from ladcp.export.netcdf import write_cruise_nc, write_station_nc


def test_station_nc_roundtrip(synth_export, tmp_path):
    path = tmp_path / "MORIA-80.nc"
    write_station_nc(synth_export, str(path))
    ds = xr.open_dataset(path)
    try:
        assert "depth" in ds.coords
        assert {"u", "v", "uerr", "nvel"} <= set(ds.data_vars)
        assert ds["u"].attrs["units"] == "m s-1"
        assert ds["depth"].attrs["positive"] == "down"
        # mixed-grid arrays get their own depth dims
        assert "depth_bt" in ds.coords and "u_bt" in ds.data_vars
        assert "depth_sadcp" in ds.coords
        # global metadata
        assert ds.attrs["station"] == "MORIA-80"
        assert ds.attrs["solver"] == "inverse"
        assert ds.attrs["Conventions"] == "CF-1.8"
    finally:
        ds.close()


def test_cruise_nc_union_grid(synth_exports, tmp_path):
    path = tmp_path / "MORIA_ladcp.nc"
    write_cruise_nc(synth_exports, str(path), cruise="MORIA")
    ds = xr.open_dataset(path)
    try:
        assert list(ds["station"].values) == ["MORIA-79", "MORIA-80"]
        assert ds["u"].dims == ("station", "depth")
        # union grid spans 10..220 m on a 10 m step minus the shared NaN bin (30 m) = 21
        assert ds.sizes["depth"] == 21
        row79 = ds["u"].sel(station="MORIA-79").values
        assert np.isfinite(row79[:17]).all()       # the shallow station fills its 17 bins
        assert np.isnan(row79[17:]).all()          # and is NaN-padded past 180 m
        assert ds["latitude"].attrs["units"] == "degrees_north"
        assert ds.attrs["cruise"] == "MORIA"
    finally:
        ds.close()
