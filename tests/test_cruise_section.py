"""Cruise velocity section product (``ladcp-section``).

Covers the vendored objective-analysis gridder's honesty masking, cross/along-track rotation,
the linear fallback, the cruise-NetCDF loader round-trip, and an end-to-end figure build on a
synthetic 5-station transect.
"""

from __future__ import annotations

import numpy as np
import pytest

from ladcp.plots.cruise_section import (
    CruiseSection,
    cruise_section_figure,
    load_cruise_section,
    rotate_cross_along,
)
from ladcp.plots.section_grid import auto_oa_params, grid_linear, grid_oa


def _profiles(x_st, z, vals):
    return [{"dep": z, "val": vals[j]} for j in range(len(x_st))]


def test_grid_oa_masks_below_seabed_and_above_surface():
    z = np.linspace(0, 200, 21)
    x_st = np.array([0.0, 10.0, 20.0])
    # each station sampled only 40-160 m; seabed at 180 m everywhere
    vals = []
    for _ in x_st:
        col = np.full(z.size, np.nan)
        col[(z >= 40) & (z <= 160)] = 0.2
        vals.append(col)
    x_grid = np.linspace(0, 20, 30)
    bathy = np.full_like(x_grid, 180.0)
    lh, lv = auto_oa_params(x_st, 200.0)
    Z = grid_oa(_profiles(x_st, z, vals), x_grid, z, x_st, bathy, lh, lv)
    assert Z.shape == (z.size, x_grid.size)
    # above shallowest observation (40 m) -> NaN
    assert np.all(np.isnan(Z[z < 35]))
    # below seabed (180 m) -> NaN
    assert np.all(np.isnan(Z[z > 185]))
    # sampled core -> finite and near the injected value
    core = Z[(z >= 60) & (z <= 140)]
    assert np.nanmean(core) == pytest.approx(0.2, abs=0.02)


def test_grid_oa_blanks_far_from_any_cast():
    z = np.linspace(0, 100, 11)
    x_st = np.array([0.0, 2.0])          # both casts near x=0..2; Lh small
    vals = [np.full(z.size, 0.1), np.full(z.size, 0.1)]
    x_grid = np.linspace(0, 100, 50)     # extends far past the casts
    bathy = np.full_like(x_grid, 120.0)
    Z = grid_oa(_profiles(x_st, z, vals), x_grid, z, x_st, bathy, lh=1.0, lv=20.0)
    # nodes ~50 km away (>> 3*Lh) must be NaN
    assert np.all(np.isnan(Z[:, x_grid > 20]))
    # nodes at the casts are finite
    assert np.isfinite(Z[5, 0])


def test_rotate_cross_along_due_east():
    # heading 90 deg (due east): along == u (east), cross == v (north)
    u = np.array([[0.3, 0.3]])
    v = np.array([[0.1, 0.1]])
    cross, along = rotate_cross_along(u, v, np.array([90.0]))
    assert along == pytest.approx(0.3, abs=1e-9)
    assert cross == pytest.approx(0.1, abs=1e-9)


def test_rotate_cross_along_due_north():
    # heading 0 deg (due north): along == v (north), cross == -u (left of north = west)
    u = np.array([[0.2]])
    v = np.array([[0.4]])
    cross, along = rotate_cross_along(u, v, np.array([0.0]))
    assert along == pytest.approx(0.4, abs=1e-9)
    assert cross == pytest.approx(-0.2, abs=1e-9)


def test_grid_linear_no_horizontal_extrapolation():
    z = np.linspace(0, 100, 11)
    x_st = np.array([10.0, 20.0])
    vals = [np.full(z.size, 0.1), np.full(z.size, 0.3)]
    x_grid = np.linspace(0, 30, 31)
    bathy = np.full_like(x_grid, 120.0)
    Z = grid_linear(_profiles(x_st, z, vals), x_grid, z, x_st, bathy)
    # outside the [10, 20] station span -> NaN (no extrapolation)
    assert np.all(np.isnan(Z[:, x_grid < 10]))
    assert np.all(np.isnan(Z[:, x_grid > 20]))
    # midpoint interpolates between 0.1 and 0.3
    mid = int(np.argmin(np.abs(x_grid - 15.0)))
    assert np.nanmean(Z[:, mid]) == pytest.approx(0.2, abs=0.02)


def _synthetic_section(ns=5, nz=25):
    z = np.linspace(0, 240, nz)
    lon = np.linspace(3.0, 3.4, ns)        # eastward transect
    lat = np.full(ns, 42.3)
    u = np.tile(np.linspace(-0.2, 0.2, nz), (ns, 1))
    v = np.tile(np.linspace(0.1, -0.1, nz), (ns, 1))
    # blank below a sloping seabed
    bottom = np.linspace(150, 230, ns)
    for j in range(ns):
        u[j, z > bottom[j]] = np.nan
        v[j, z > bottom[j]] = np.nan
    return CruiseSection(
        stations=np.array([f"st{j:02d}" for j in range(ns)]),
        depth=z, u=u, v=v, uerr=np.full((ns, nz), 0.02),
        lat=lat, lon=lon, bottom_depth=bottom)


def test_loader_roundtrip(tmp_path):
    import xarray as xr

    sec = _synthetic_section()
    ds = xr.Dataset(
        {"u": (("station", "depth"), sec.u),
         "v": (("station", "depth"), sec.v),
         "uerr": (("station", "depth"), sec.uerr),
         "bottom_depth": ("station", sec.bottom_depth)},
        coords={"station": sec.stations, "depth": sec.depth,
                "latitude": ("station", sec.lat), "longitude": ("station", sec.lon)})
    p = tmp_path / "TEST_ladcp.nc"
    ds.to_netcdf(p)
    got = load_cruise_section(str(p))
    assert list(got.stations) == list(sec.stations)
    np.testing.assert_allclose(got.bottom_depth, sec.bottom_depth)
    # also resolve from a cruise dir holding exports/*_ladcp.nc
    (tmp_path / "exports").mkdir()
    ds.to_netcdf(tmp_path / "exports" / "TEST_ladcp.nc")
    assert load_cruise_section(str(tmp_path)).stations.size == sec.stations.size


def test_figure_builds_panels():
    import matplotlib
    matplotlib.use("Agg")
    sec = _synthetic_section()
    fig = cruise_section_figure(sec, component="both")
    assert len(fig.axes) >= 4           # 4 component panels (+ colorbar axes)
    fig2 = cruise_section_figure(sec, component="cross-along", gridding="linear")
    assert fig2 is not None
