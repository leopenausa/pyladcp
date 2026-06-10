"""CODAS shipboard-ADCP reader (ladcp.io.sadcp_codas).

The reader reshapes a CODAS short-form ("contour") NetCDF onto the SadcpDataset
interface: decimal-day time decode, 2-D depth collapsed to the per-bin median,
pflag-based blanking, and velocities passed through untouched (CODAS u/v are
already absolute ocean velocity). Tests run on a small hand-built NetCDF; an
optional real-data test exercises the MORIA OS150 product when present.
"""

from __future__ import annotations

import pathlib

import numpy as np
import pytest

from ladcp.io.sadcp_codas import read_codas_nc, resolve_codas_nc
from ladcp.io.sadcp_vmdas import extract_profile

N_T = 6
N_Z = 5
LAT0, LON0 = 62.0, -11.0
# decimal days for 2025-10-03T06:30:00 onward, one ensemble per 2 min
_T0_DAYS = (np.datetime64("2025-10-03T06:30:00") - np.datetime64("2025-01-01")
            ) / np.timedelta64(1, "D")


def _write_codas_nc(path, *, u0=0.10, v0=-0.20, pflag_bad_cell=None,
                    depth_jitter=0.0, sonar="os150nb"):
    """A miniature CODAS contour file: constant absolute velocity, optional
    per-cell pflag editing and per-ensemble depth-grid jitter."""
    from netCDF4 import Dataset

    with Dataset(path, "w", format="NETCDF3_CLASSIC") as nc:
        nc.sonar = sonar
        nc.cruise_id = "SYN"
        nc.createDimension("time", N_T)
        nc.createDimension("depth_cell", N_Z)
        nc.createDimension("num_configs", 1)

        t = nc.createVariable("time", "f8", ("time",))
        t.units = "days since 2025-01-01 00:00:00"
        t[:] = _T0_DAYS + np.arange(N_T) * (120.0 / 86400.0)

        nc.createVariable("lat", "f8", ("time",))[:] = np.full(N_T, LAT0)
        nc.createVariable("lon", "f8", ("time",))[:] = np.full(N_T, LON0)

        d = nc.createVariable("depth", "f4", ("time", "depth_cell"))
        grid = 17.0 + 8.0 * np.arange(N_Z)
        d[:] = grid[None, :] + depth_jitter * np.linspace(-1, 1, N_T)[:, None]

        u = nc.createVariable("u", "f4", ("time", "depth_cell"))
        v = nc.createVariable("v", "f4", ("time", "depth_cell"))
        u[:] = np.full((N_T, N_Z), u0)
        v[:] = np.full((N_T, N_Z), v0)

        pf = nc.createVariable("pflag", "i1", ("time", "depth_cell"))
        flags = np.zeros((N_T, N_Z), "i1")
        if pflag_bad_cell is not None:
            flags[:, pflag_bad_cell] = 2          # "below bottom" editing flag
        pf[:] = flags

        nc.createVariable("pg", "i1", ("time", "depth_cell"))[:] = \
            np.full((N_T, N_Z), 100, "i1")
        nc.createVariable("transducer_depth", "f4", ("num_configs",))[:] = [5.0]
    return path


def test_read_codas_nc_decodes_time_grid_and_velocity(tmp_path):
    f = _write_codas_nc(tmp_path / "os150nb.nc")
    ds = read_codas_nc(f)
    assert ds.file_type == "CODAS" and ds.freq_khz == 150 and ds.n_ens == N_T
    assert ds.time[0] == np.datetime64("2025-10-03T06:30:00", "ns")
    assert ds.time[-1] - ds.time[0] == np.timedelta64((N_T - 1) * 120, "s")
    np.testing.assert_allclose(ds.depth, 17.0 + 8.0 * np.arange(N_Z), atol=1e-5)
    assert ds.u.shape == (N_Z, N_T)               # [z, t] like the VmDAS ingester
    # CODAS u/v are already absolute -> passed through, no ship velocity added
    np.testing.assert_allclose(ds.u, 0.10, atol=1e-6)
    np.testing.assert_allclose(ds.v, -0.20, atol=1e-6)
    assert ds.transducer_depth == 5.0


def test_depth_jitter_collapses_to_median_grid(tmp_path):
    f = _write_codas_nc(tmp_path / "os150nb.nc", depth_jitter=0.02)
    ds = read_codas_nc(f)                          # sound-speed wobble ~ +-2 cm
    np.testing.assert_allclose(ds.depth, 17.0 + 8.0 * np.arange(N_Z), atol=0.05)
    assert np.all(np.diff(ds.depth) > 0)


def test_pflag_editing_blanks_cells(tmp_path):
    f = _write_codas_nc(tmp_path / "os150nb.nc", pflag_bad_cell=3)
    ds = read_codas_nc(f)
    assert np.all(np.isnan(ds.u[3])) and np.all(np.isnan(ds.v[3]))
    assert np.all(np.isfinite(ds.u[0]))            # unedited neighbours kept


def test_extract_profile_runs_on_codas_dataset(tmp_path):
    f = _write_codas_nc(tmp_path / "os150nb.nc")
    ds = read_codas_nc(f)
    t0, t1 = ds.time[0], ds.time[-1]
    sv = extract_profile(ds, time_start=t0, time_end=t1, lat=LAT0, lon=LON0)
    assert sv is not None and sv.shape == (N_Z, 4)
    np.testing.assert_allclose(sv[:, 1], 0.10, atol=1e-6)
    np.testing.assert_allclose(sv[:, 2], -0.20, atol=1e-6)
    # far-away station still rejected by the position gate
    assert extract_profile(ds, time_start=t0, time_end=t1, lat=55.0, lon=-3.0) is None


def test_resolve_codas_nc_file_dir_and_contour(tmp_path):
    (tmp_path / "contour").mkdir()
    f = _write_codas_nc(tmp_path / "contour" / "os150nb.nc")
    assert resolve_codas_nc(f) == f                          # the file itself
    assert resolve_codas_nc(tmp_path / "contour") == f       # dir with one .nc
    assert resolve_codas_nc(tmp_path) == f                   # processing dir
    with pytest.raises(FileNotFoundError):
        resolve_codas_nc(tmp_path / "missing")
    _write_codas_nc(tmp_path / "contour" / "os75bb.nc")      # ambiguity -> error
    with pytest.raises(FileNotFoundError):
        resolve_codas_nc(tmp_path / "contour")


# --------------------------------------------------------------------------- #
# real-data test (skipped unless the MORIA CODAS product is present)
# --------------------------------------------------------------------------- #
_REAL = pathlib.Path("/home/leo/Cruises/MORIA/data/pyladcp_data/MORIA/codas/"
                     "os150nb_sta/contour/os150nb.nc")


@pytest.mark.skipif(not _REAL.exists(), reason="MORIA CODAS product not present")
def test_real_moria_codas_product_reads_and_windows():
    ds = read_codas_nc(_REAL)
    assert ds.freq_khz == 150 and ds.n_ens > 15_000
    assert ds.time[0].astype("datetime64[s]") == np.datetime64("2025-09-15T10:24:10")
    assert np.all(np.diff(ds.depth) > 0) and 16 < ds.depth[0] < 18
    # MORIA-80 cast window must yield a usable constraint profile
    sv = extract_profile(ds, time_start="2025-10-03T06:27:12",
                         time_end="2025-10-03T07:21:39", lat=62.16135, lon=-11.53085)
    assert sv is not None and sv.shape[0] > 20
    assert np.nanmax(np.abs(sv[:, 1:3])) < 1.5    # ocean-scale velocities
